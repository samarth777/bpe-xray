"""Act III (Breakthrough v2): BPE-Invariant Weight-Space Word Retina + Mid-Layer Visual Cortex.

Pure weight-space gated linear recurrence compiled directly from tokenizer.json:
  - ZERO Python string parsing, ZERO tokenizer calls at inference, ZERO tool syntax!
  - Exact BPE-tokenization-invariant pooling across subword boundaries (`str+aw+berry` == `s+traw+ber+ry`)
  - Exact cross-BPE-boundary double-letter detector (`E_last[t-1] · E_first[t]`)
  - Exact ordered character grid (`[16, 26]`) via compiled ReLU slot placement
  - Exact bilinear query-letter match, exact k-th letter ReLU readout, exact anagram L1 diff
  - Bolted onto mid-to-late layers (16, 20, 24) of a 100% FROZEN Qwen3-0.6B with exact
    zero-init output projections -> bit-for-bit identical at step 0!
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_tables import CharWeightBank
from xray_alu import relu_eq_int


def _relu_int_buckets(x: torch.Tensor, max_k: int) -> torch.Tensor:
    """Exact ReLU integer-bucket encoding for integer values 0..max_k."""
    grid = torch.arange(max_k + 1, device=x.device, dtype=x.dtype)
    return F.relu(1.0 - torch.abs(x.unsqueeze(-1) - grid))


class WordRetinaScanner(nn.Module):
    """Pure weight-space gated linear recurrence compiling sub-token tables into
    BPE-tokenization-invariant word/span-level sensory vectors.

    Produces per-token feature vector S_t [B, T, DIM=623]:
      1. `w_hist`            (26)  : exact letter counts a..z of target span
      2. `w_len_buckets`     (25)  : exact ReLU integer one-hot of alpha length 0..24
      3. `w_len_scalar`      (1)   : w_len / 20.0
      4. `w_dbl`             (1)   : exact double-letter flag (internal OR cross-BPE!)
      5. `matched_cnt_bkt`   (21)  : exact ReLU integer one-hot of (w_hist · q_letter) in 0..20
      6. `matched_cnt_sc`    (1)   : matched_count / 10.0
      7. `matched_nth_oh`    (26)  : exact 1-hot of k-th letter (using multi-digit q_num register!)
      8. `anagram_feat`      (2)   : [exact_anagram_flag, l1_hist_diff / 10.0] vs previous span
      9. `w_first_last`      (52)  : exact first letter (26) + last letter (26)
     10. `q_letters`         (52)  : [q_letter (26), q_letter_prev (26)] (supports delete & replace!)
     11. `w_chars_flat`      (416) : exact ordered characters [0..15] x 26 (BPE-invariant!)
    Total: 623 exact invariant sensory dims!
    """

    MAX_WORD_GRID = 16
    DIM = 26 + 25 + 1 + 1 + 21 + 1 + 26 + 2 + 52 + 52 + (16 * 26)  # 623

    def __init__(self, bank: CharWeightBank, tokenizer):
        super().__init__()
        self.register_buffer("F_retina", bank.F_retina.float().clone())             # [V, 358]
        self.register_buffer("E_alpha_chars", bank.E_alpha_chars.clone())           # [V, L_max]
        self.register_buffer("char_grid", torch.arange(1, 27, dtype=torch.float32))
        self.register_buffer("pos_grid", torch.arange(self.MAX_WORD_GRID, dtype=torch.float32))
        self.register_buffer("tok_slot_grid", torch.arange(bank.max_tok_chars, dtype=torch.float32))

        V = bank.vocab_size
        E_num_val = torch.zeros(V, dtype=torch.float32)
        E_num_pow10 = torch.ones(V, dtype=torch.float32)
        E_is_num = torch.zeros(V, dtype=torch.float32)
        E_is_quote = torch.zeros(V, dtype=torch.float32)

        # Compile pure token-ID lookup for digits 0..24 (handles both single-digit Qwen3 & multi-digit tokenizers!)
        for k in range(0, 25):
            p10 = float(10 ** len(str(k)))
            for s in [str(k), " " + str(k)]:
                tids = tokenizer.encode(s, add_special_tokens=False)
                if len(tids) == 1:
                    E_num_val[tids[0]] = float(k)
                    E_num_pow10[tids[0]] = p10
                    E_is_num[tids[0]] = 1.0
            if k >= 1:
                for suf in ["st", "nd", "rd", "th"]:
                    for pre in ["", " "]:
                        tids = tokenizer.encode(f"{pre}{k}{suf}", add_special_tokens=False)
                        if len(tids) == 1:
                            E_num_val[tids[0]] = float(k)
                            E_num_pow10[tids[0]] = p10
                            E_is_num[tids[0]] = 1.0

        # Compile pure token-ID lookup for quote delimiter tokens (`'`, ` '`, `'?'`, `'.`, etc., no letters!)
        tok_len = len(tokenizer)
        decoded = tokenizer.batch_decode(
            [[i] for i in range(tok_len)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for tid, s in enumerate(decoded):
            if ("'" in s or '"' in s) and not any(c.isalpha() for c in s):
                E_is_quote[tid] = 1.0

        self.register_buffer("E_num_val", E_num_val)
        self.register_buffer("E_num_pow10", E_num_pow10)
        self.register_buffer("E_is_num", E_is_num)
        self.register_buffer("E_is_quote", E_is_quote)

        # Causal intervention hook for Optical Illusion experiments!
        self.feature_override_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    def _init_state(self, B: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            # Active quoted-span accumulator
            "in_quote": torch.zeros(B, device=device),
            "prev_has_alpha": torch.zeros(B, device=device),
            "prev_last": torch.zeros(B, 26, device=device),
            "sp_hist": torch.zeros(B, 26, device=device),
            "sp_len": torch.zeros(B, device=device),
            "sp_dbl": torch.zeros(B, device=device),
            "sp_first": torch.zeros(B, 26, device=device),
            "sp_last": torch.zeros(B, 26, device=device),
            "sp_chars": torch.zeros(B, self.MAX_WORD_GRID, 26, device=device),
            # Latched target-span registers (persists across `?\nAnswer:` and generated tokens!)
            "has_quoted": torch.zeros(B, device=device),
            "lat_hist": torch.zeros(B, 26, device=device),
            "lat_len": torch.zeros(B, device=device),
            "lat_dbl": torch.zeros(B, device=device),
            "lat_first": torch.zeros(B, 26, device=device),
            "lat_last": torch.zeros(B, 26, device=device),
            "lat_chars": torch.zeros(B, self.MAX_WORD_GRID, 26, device=device),
            # Previous multi-char quoted span (for anagram comparison!)
            "prev_hist": torch.zeros(B, 26, device=device),
            "prev_valid": torch.zeros(B, device=device),
            # Query letter registers (latest & previous 1-letter quoted spans, e.g. 'a' and 'o')
            "q_letter": torch.zeros(B, 26, device=device),
            "q_letter_prev": torch.zeros(B, 26, device=device),
            # Multi-digit numeric query register
            "q_num": torch.zeros(B, device=device),
            "prev_is_num": torch.zeros(B, device=device),
        }

    def compute_stream(
        self,
        input_ids: torch.Tensor,  # [B, T]
        prev_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = input_ids.shape
        dev = input_ids.device

        f = self.F_retina[input_ids]  # [B, T, 358]
        if self.feature_override_fn is not None:
            f = self.feature_override_fn(input_ids, f)

        hist = f[:, :, 0:26]                   # [B, T, 26]
        first_a = f[:, :, 27:53]               # [B, T, 26]
        last_a = f[:, :, 54:80]                # [B, T, 26]
        alpha_len = f[:, :, 351] * 10.0        # [B, T]
        lead_space = f[:, :, 354]              # [B, T]
        int_dbl = f[:, :, 355]                 # [B, T]
        has_alpha = (alpha_len > 0.5).float()  # [B, T]

        tok_codes = self.E_alpha_chars[input_ids].float()  # [B, T, L_max]
        tok_oh = relu_eq_int(tok_codes.unsqueeze(-1), self.char_grid.view(1, 1, 1, 26))  # [B, T, L_max, 26]

        is_num = self.E_is_num[input_ids]        # [B, T]
        num_val = self.E_num_val[input_ids]      # [B, T]
        num_pow10 = self.E_num_pow10[input_ids]  # [B, T]
        is_quote = self.E_is_quote[input_ids]    # [B, T]

        st = self._init_state(B, dev) if prev_state is None else {k: v.clone() for k, v in prev_state.items()}
        outputs: List[torch.Tensor] = []

        for t in range(T):
            ha = has_alpha[:, t]
            ls = lead_space[:, t]
            al = alpha_len[:, t]
            h_t = hist[:, t]
            fa_t = first_a[:, t]
            la_t = last_a[:, t]
            idbl_t = int_dbl[:, t]
            q_t = is_quote[:, t]

            # 1. Update multi-digit numeric query register (e.g. "1" then "2" -> 12!)
            is_n = is_num[:, t]
            cont_num = is_n * st["prev_is_num"]
            new_qnum = cont_num * (st["q_num"] * num_pow10[:, t] + num_val[:, t]) + (is_n * (1.0 - cont_num)) * num_val[:, t]
            st["q_num"] = is_n * new_qnum + (1.0 - is_n) * st["q_num"]
            st["prev_is_num"] = is_n

            # 2. Quote state transitions (pure algebraic gates!)
            closing_q = q_t * st["in_quote"]
            opening_q = q_t * (1.0 - st["in_quote"])

            # On closing quote:
            # (a) If span had exactly 1 letter -> push into q_letter register (and shift old to q_letter_prev)!
            is_one_let = closing_q * relu_eq_int(st["sp_len"], torch.ones_like(st["sp_len"]))
            m1 = is_one_let.unsqueeze(-1)
            st["q_letter_prev"] = m1 * st["q_letter"] + (1.0 - m1) * st["q_letter_prev"]
            st["q_letter"] = m1 * st["sp_hist"] + (1.0 - m1) * st["q_letter"]

            # (b) If span had >= 2 letters -> shift old lat_hist to prev_hist, and latch new target span!
            is_multi_let = closing_q * (st["sp_len"] >= 2.0).float()
            mm = is_multi_let.unsqueeze(-1)
            st["prev_hist"] = (mm * st["has_quoted"].unsqueeze(-1)) * st["lat_hist"] + (1.0 - mm * st["has_quoted"].unsqueeze(-1)) * st["prev_hist"]
            st["prev_valid"] = torch.clamp(st["prev_valid"] + is_multi_let * st["has_quoted"], 0.0, 1.0)
            st["has_quoted"] = torch.clamp(st["has_quoted"] + is_multi_let, 0.0, 1.0)

            st["lat_hist"] = mm * st["sp_hist"] + (1.0 - mm) * st["lat_hist"]
            st["lat_len"] = is_multi_let * st["sp_len"] + (1.0 - is_multi_let) * st["lat_len"]
            st["lat_dbl"] = is_multi_let * st["sp_dbl"] + (1.0 - is_multi_let) * st["lat_dbl"]
            st["lat_first"] = mm * st["sp_first"] + (1.0 - mm) * st["lat_first"]
            st["lat_last"] = mm * st["sp_last"] + (1.0 - mm) * st["lat_last"]
            st["lat_chars"] = mm.unsqueeze(-1) * st["sp_chars"] + (1.0 - mm.unsqueeze(-1)) * st["lat_chars"]

            # Update in_quote and reset span accumulator on opening/closing quote
            st["in_quote"] = st["in_quote"] * (1.0 - closing_q) + opening_q
            reset_sp = torch.clamp(opening_q + closing_q, 0.0, 1.0)
            keep_sp = 1.0 - reset_sp
            k1 = keep_sp.unsqueeze(-1)
            st["sp_hist"] = k1 * st["sp_hist"]
            st["sp_len"] = keep_sp * st["sp_len"]
            st["sp_dbl"] = keep_sp * st["sp_dbl"]
            st["sp_first"] = k1 * st["sp_first"]
            st["sp_last"] = k1 * st["sp_last"]
            st["sp_chars"] = k1.unsqueeze(-1) * st["sp_chars"]

            # 3. Accumulate inside active quoted span!
            active = st["in_quote"] * ha  # [B]
            a1 = active.unsqueeze(-1)
            # Cross-BPE double-letter detector (only within same word: no leading space!)
            same_word = active * (1.0 - ls) * st["prev_has_alpha"]
            cross_dbl = same_word * (st["prev_last"] * fa_t).sum(dim=-1)

            # Exact ordered character placement at offset `sp_len` via ReLU slot matching
            start_off = st["sp_len"]  # [B]
            place_w = relu_eq_int(
                self.pos_grid.view(1, self.MAX_WORD_GRID, 1),
                start_off.view(B, 1, 1) + self.tok_slot_grid.view(1, 1, -1),
            )  # [B, 16, L_max]
            placed = torch.einsum("bgm,bmc->bgc", place_w, tok_oh[:, t])

            is_first_in_span = active * relu_eq_int(st["sp_len"], torch.zeros_like(st["sp_len"]))
            st["sp_first"] = is_first_in_span.unsqueeze(-1) * fa_t + (1.0 - is_first_in_span.unsqueeze(-1)) * st["sp_first"]
            st["sp_last"] = a1 * la_t + (1.0 - a1) * st["sp_last"]
            st["sp_hist"] = st["sp_hist"] + a1 * h_t
            st["sp_len"] = st["sp_len"] + active * al
            st["sp_dbl"] = torch.clamp(st["sp_dbl"] + active * (idbl_t + cross_dbl), 0.0, 1.0)
            st["sp_chars"] = st["sp_chars"] + a1.unsqueeze(-1) * placed

            st["prev_has_alpha"] = active
            st["prev_last"] = a1 * la_t

            # 4. Compute exact invariant readouts from latched target span!
            w_hist = st["lat_hist"]
            w_len = st["lat_len"]
            w_dbl = st["lat_dbl"]
            w_first = st["lat_first"]
            w_last = st["lat_last"]
            w_chars = st["lat_chars"]

            matched_cnt = (w_hist * st["q_letter"]).sum(dim=-1)  # [B]
            k_sel = relu_eq_int(self.pos_grid.view(1, self.MAX_WORD_GRID), (st["q_num"] - 1.0).view(B, 1))
            matched_nth = torch.einsum("bg,bgc->bc", k_sel, w_chars)  # [B, 26]

            l1_diff = torch.abs(w_hist - st["prev_hist"]).sum(dim=-1)
            is_anagram = st["prev_valid"] * F.relu(1.0 - l1_diff)

            feat_t = torch.cat(
                [
                    w_hist / 4.0,                                # 26
                    _relu_int_buckets(w_len, 24),                # 25
                    (w_len / 20.0).unsqueeze(-1),                # 1
                    w_dbl.unsqueeze(-1),                         # 1
                    _relu_int_buckets(matched_cnt, 20),          # 21
                    (matched_cnt / 10.0).unsqueeze(-1),          # 1
                    matched_nth,                                 # 26
                    is_anagram.unsqueeze(-1),                    # 1
                    (l1_diff / 10.0).unsqueeze(-1),              # 1
                    w_first,                                     # 26
                    w_last,                                      # 26
                    st["q_letter"],                              # 26
                    st["q_letter_prev"],                         # 26
                    w_chars.reshape(B, self.MAX_WORD_GRID * 26), # 416
                ],
                dim=-1,
            )  # [B, 623]
            outputs.append(feat_t)

        S = torch.stack(outputs, dim=1)  # [B, T, 623]
        new_state = {k: v.detach() for k, v in st.items()}
        return S, new_state


class LatentIntuitionTap(nn.Module):
    """Mid-layer bilinear sensory readout injected into a FROZEN host layer h_l.
    Exact ZERO-initialized `o_proj` -> bit-for-bit identical at step 0!
    """

    def __init__(
        self,
        hidden_size: int,
        sensory_dim: int = WordRetinaScanner.DIM,
        bottleneck_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h_norm = nn.RMSNorm(hidden_size)
        self.h_gate = nn.Linear(hidden_size, bottleneck_dim, bias=True)
        self.s_proj = nn.Linear(sensory_dim, bottleneck_dim, bias=False)
        self.mix = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)
        self.h_gate2 = nn.Linear(hidden_size, bottleneck_dim, bias=True)
        self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, h: torch.Tensor, S: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if scale == 0.0:
            return torch.zeros_like(h)
        h_f = self.h_norm(h.float())
        S_f = S.float()
        z1 = F.silu(self.h_gate(h_f)) * self.s_proj(S_f)
        z2 = F.silu(self.h_gate2(h_f)) * self.mix(z1)
        delta = self.o_proj(self.drop(z2)) * float(scale)
        return delta.to(h.dtype)


class WordRetinaLM(nn.Module):
    """Bolts the BPE-Invariant Word Retina + Mid-Layer Latent Intuition Taps onto a FROZEN LM."""

    def __init__(
        self,
        model: nn.Module,
        bank: CharWeightBank,
        tokenizer,
        tap_layers: Sequence[int] = (16, 20, 24),
        bottleneck_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.tap_layers = tuple(tap_layers)
        hidden_size = model.config.hidden_size
        dev = next(model.parameters()).device

        self.scanner = WordRetinaScanner(bank, tokenizer).to(dev)
        self.taps = nn.ModuleDict({
            str(l): LatentIntuitionTap(
                hidden_size=hidden_size,
                sensory_dim=WordRetinaScanner.DIM,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
            ).to(dev)
            for l in self.tap_layers
        })

        self.retina_scale: float = 1.0
        self._cur_S: Optional[torch.Tensor] = None
        self._caching: bool = False
        self._scan_state: Optional[Dict[str, torch.Tensor]] = None
        self._handles: List = []
        self.attach()

    def set_retina_scale(self, scale: float):
        self.retina_scale = float(scale)

    def set_feature_override(self, fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]):
        self.scanner.feature_override_fn = fn

    def attach(self):
        if self._handles:
            return self
        embed = self.model.get_input_embeddings()

        def _embed_hook(_mod, inp, _out):
            input_ids = inp[0]
            if self._caching:
                S, self._scan_state = self.scanner.compute_stream(input_ids, prev_state=self._scan_state)
            else:
                S, _ = self.scanner.compute_stream(input_ids, prev_state=None)
            self._cur_S = S

        self._handles.append(embed.register_forward_hook(_embed_hook))

        layers = self.model.model.layers
        for l_idx in self.tap_layers:
            key = str(l_idx)
            tap = self.taps[key]

            def _make_hook(t_mod=tap):
                def _layer_hook(_mod, _inp, out):
                    if self.retina_scale == 0.0 or self._cur_S is None:
                        return out
                    is_tuple = isinstance(out, tuple)
                    h = out[0] if is_tuple else out
                    delta = t_mod(h, self._cur_S, scale=self.retina_scale)
                    h_new = h + delta
                    if is_tuple:
                        return (h_new,) + out[1:]
                    return h_new
                return _layer_hook

            self._handles.append(layers[l_idx].register_forward_hook(_make_hook()))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    def forward(self, *args, **kwargs):
        self._caching = False
        self._scan_state = None
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self._caching = True
        self._scan_state = None
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            self._caching = False
            self._scan_state = None
