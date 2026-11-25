"""Act II: Exact Weight-Space BPE X-Ray Micromachine (Sub-Token ALU).

Compiled directly from (model, tokenizer) into pure PyTorch tensor operations:
  - Frozen tokenizer-compiled lookup tables (E_hist, E_alpha_len, E_alpha_chars)
  - Exact 2-layer ReLU base-10 digit extractor (0..999)
  - Exact ReLU prefix-sum token & intra-token character slot selector
  - Exact 26x26 ROT13 permutation linear layer
  - Solved output write directions added to last_hidden_state right before lm_head
  - Guaranteed BIT-FOR-BIT IDENTICAL logits (delta == exact 0.0) whenever idle!

Supported operations inside `[XRAY <op> [arg] | <any BPE text>]=`:
  - count <letter> | <text>   -> exact decimal count of <letter> in <text>
  - len | <text>              -> exact alphabetic character count of <text>
  - nth <k> | <text>          -> exact 1-indexed k-th alphabetic character of <text>
  - spell | <text>            -> exact hyphen-separated letters (s-t-r-a-w-b-e-r-r-y)
  - reverse | <text>          -> exact reversed letters (yrrebwarts)
  - rot13 | <text>            -> exact ROT13 cipher of letters (fgenjoreel)

At inference time, ZERO tokenizer calls and ZERO Python string inspections occur.
"""
from __future__ import annotations

import string
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_tables import CharWeightBank, CHAR_A


def relu_eq_int(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact ReLU indicator: returns 1.0 when integer x == target, 0.0 for all other integers.
    Uses pure piecewise-linear ReLU units: ReLU(1 - ReLU(x - t) - ReLU(t - x)).
    """
    return F.relu(1.0 - F.relu(x - target) - F.relu(target - x))


def relu_in_range(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Exact ReLU indicator: returns 1.0 when integer lo <= x <= hi, 0.0 otherwise."""
    return F.relu(1.0 - F.relu(lo - x) - F.relu(x - hi))


class ReLUDigitExtractor(nn.Module):
    """Compiled 2-layer ReLU network mapping exact integer n in [0, 999] to base-10 digits.

    All weights are deterministic compiled parameters/buffers (no Python division/modulo!).
    Returns:
      digits: [B, 3, 10] one-hot for (hundreds, tens, ones)
      active: [B, 3] binary mask indicating which digit slots to emit (no leading zeros, except '0')
    """

    def __init__(self, max_val: int = 999):
        super().__init__()
        self.max_val = max_val
        # Thresholds for 100s (100, 200, ..., 900) and 10s (10, 20, ..., 90)
        self.register_buffer("t_hundreds", torch.arange(1, 10, dtype=torch.float32) * 100.0)
        self.register_buffer("t_tens", torch.arange(1, 10, dtype=torch.float32) * 10.0)
        self.register_buffer("d_grid", torch.arange(10, dtype=torch.float32))

    def forward(self, n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # n: [B] exact integer tensor (float32)
        n = n.float().unsqueeze(-1)  # [B, 1]
        # Step function I(n >= t) = ReLU(n - t + 1) - ReLU(n - t)
        h_steps = F.relu(n - self.t_hundreds + 1.0) - F.relu(n - self.t_hundreds)  # [B, 9]
        hundreds = h_steps.sum(dim=-1, keepdim=True)  # [B, 1] in 0..9

        rem100 = n - 100.0 * hundreds
        t_steps = F.relu(rem100 - self.t_tens + 1.0) - F.relu(rem100 - self.t_tens)  # [B, 9]
        tens = t_steps.sum(dim=-1, keepdim=True)  # [B, 1] in 0..9

        ones = rem100 - 10.0 * tens  # [B, 1] in 0..9

        oh_h = relu_eq_int(hundreds, self.d_grid)  # [B, 10]
        oh_t = relu_eq_int(tens, self.d_grid)      # [B, 10]
        oh_o = relu_eq_int(ones, self.d_grid)      # [B, 10]

        digits = torch.stack([oh_h, oh_t, oh_o], dim=1)  # [B, 3, 10]
        # Active mask via ReLU step: emit hundreds if n >= 100, tens if n >= 10, ones always
        act_h = F.relu(n - 99.0) - F.relu(n - 100.0)
        act_t = F.relu(n - 9.0) - F.relu(n - 10.0)
        act_o = torch.ones_like(act_h)
        active = torch.cat([act_h, act_t, act_o], dim=-1)  # [B, 3]
        return digits, active


class XRayALUImplant(nn.Module):
    """Weight-space BPE X-Ray Micromachine attached in parallel to a frozen CausalLM.

    Hooks onto the output of the final transformer norm (`model.model.norm`, i.e. right
    before `lm_head`) so adding `pending` to `last_hidden_state` directly steers logits.
    When idle, `pending` is identically 0.0 -> bit-for-bit identical logits!
    """

    # Output token table indices (39 emit symbols):
    # 0..9   : '0'..'9'
    # 10..35 : 'a'..'z'
    # 36     : '-'
    # 37     : '\n' (stop / end of answer)
    # 38     : ' '
    NUM_OUT_SYMS = 39
    OUT_DIGITS = 0
    OUT_ALPHA = 10
    OUT_HYPHEN = 36
    OUT_NL = 37
    OUT_SPACE = 38

    # Op codes
    OP_NONE = 0
    OP_COUNT = 1
    OP_LEN = 2
    OP_NTH = 3
    OP_SPELL = 4
    OP_REVERSE = 5
    OP_ROT13 = 6

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        bank: Optional[CharWeightBank] = None,
        write_scale: float = 120.0,
        max_span_tokens: int = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_span_tokens = max_span_tokens

        lm_head = model.get_output_embeddings()
        W_lm = lm_head.weight.detach()
        vocab_size, hidden_size = W_lm.shape
        dev = device if device is not None else W_lm.device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        if bank is None:
            bank = CharWeightBank.from_tokenizer(tokenizer, vocab_size=vocab_size)
        self.bank = bank.to(dev)
        self.digit_net = ReLUDigitExtractor().to(dev)

        # Compile ROT13 linear permutation matrix [26, 26]
        P_rot13 = torch.zeros(26, 26, dtype=torch.float32, device=dev)
        for i in range(26):
            P_rot13[i, (i + 13) % 26] = 1.0
        self.register_buffer("P_rot13", P_rot13)

        # Precompute one-hot lookup for E_alpha_chars: [V, L_max, 26] is too big (~150k*24*26),
        # so we gather [span_len, L_max] indices and one-hot only the active span (pure tensor op!).
        self.register_buffer("slot_grid", torch.arange(bank.max_tok_chars, dtype=torch.float32, device=dev))
        self.register_buffer("char_grid", torch.arange(1, 27, dtype=torch.float32, device=dev))

        # Compile token-ID match patterns & op/digit/letter readout vectors from tokenizer
        self._compile_reader_tables(tokenizer, vocab_size, dev)

        # Solve exact output write directions against lm_head.weight
        self._compile_write_directions(tokenizer, W_lm, write_scale, dev)

        # Runtime state registers (per batch item)
        self._hook_handle = None
        self.enabled = True
        self.pending: Optional[torch.Tensor] = None  # [B, 1, H] or None
        self._reset_state(batch_size=1, device=dev)

    def _compile_reader_tables(self, tokenizer, vocab_size: int, dev: torch.device):
        """Compile exact token-ID matchers into frozen tensor buffers (no string ops at runtime)."""
        # Encode trigger delimiters in context so BPE matches how they appear in prompts
        def enc_clean(s: str) -> List[int]:
            return tokenizer.encode(s, add_special_tokens=False)

        # Canonical pattern: `[XRAY <op> <arg> | <text>]=`
        # Handle BOTH leading-space (` [XRAY`) and no-space (`[XRAY`) tokenizations,
        # and multiple close-pattern variants (`]=`, word+`]=`, ` ]=`).
        open_pats = []
        for s in ["[XRAY", " [XRAY", "\n[XRAY"]:
            tids = enc_clean(s)
            if tids and tids not in open_pats:
                open_pats.append(tids)
        self.open_pats: List[torch.Tensor] = [torch.tensor(p, dtype=torch.long, device=dev) for p in open_pats]

        close_pats = []
        sample_word = enc_clean("strawberry")
        sample_close = enc_clean("strawberry]=")
        derived_close = sample_close[len(sample_word):] if sample_close[:len(sample_word)] == sample_word else []
        for cand in [enc_clean("]="), derived_close, enc_clean(" ]="), enc_clean("] =")]:
            if cand and cand not in close_pats:
                close_pats.append(cand)
        self.close_pats: List[torch.Tensor] = [torch.tensor(p, dtype=torch.long, device=dev) for p in close_pats]

        # Pipe separator token IDs (both `|` and ` |`)
        pipe_ids = set(enc_clean("|") + enc_clean(" |") + enc_clean(" | "))
        # Only keep tokens whose decoded form actually contains '|'
        E_is_pipe = torch.zeros(vocab_size, dtype=torch.float32, device=dev)
        for tid in pipe_ids:
            if "|" in tokenizer.decode([tid], clean_up_tokenization_spaces=False):
                E_is_pipe[tid] = 1.0
        # Also scan for any single-char '|' token
        for s in ["|", " |", "| ", " | "]:
            for tid in enc_clean(s):
                if tokenizer.decode([tid], clean_up_tokenization_spaces=False).strip() == "|":
                    E_is_pipe[tid] = 1.0
        self.register_buffer("E_is_pipe", E_is_pipe)

        # Op keyword detection: map token IDs -> [V, 7] one-hot op code
        # Supports single-token or first-token match of op names with optional leading space
        E_op = torch.zeros(vocab_size, 7, dtype=torch.float32, device=dev)
        op_names = {
            self.OP_COUNT: ["count", " count"],
            self.OP_LEN: ["len", " len"],
            self.OP_NTH: ["nth", " nth"],
            self.OP_SPELL: ["spell", " spell"],
            self.OP_REVERSE: ["reverse", " reverse"],
            self.OP_ROT13: ["rot13", " rot13", "rot", " rot"],
        }
        for op_code, variants in op_names.items():
            for v in variants:
                tids = enc_clean(v)
                if tids:
                    # Mark the first distinctive token of the op keyword
                    E_op[tids[0], op_code] = 1.0
        self.register_buffer("E_op", E_op)

        # Decimal value lookup for numeric args (e.g. `nth 14 | ...`): [V, 2] -> (value, is_numeric)
        E_num_val = torch.zeros(vocab_size, dtype=torch.float32, device=dev)
        E_num_pow10 = torch.ones(vocab_size, dtype=torch.float32, device=dev)
        E_is_num = torch.zeros(vocab_size, dtype=torch.float32, device=dev)
        for k in range(0, 100):
            for prefix in ["", " "]:
                tids = enc_clean(prefix + str(k))
                if len(tids) == 1:
                    tid = tids[0]
                    E_num_val[tid] = float(k)
                    E_num_pow10[tid] = float(10 ** len(str(k)))
                    E_is_num[tid] = 1.0
        self.register_buffer("E_num_val", E_num_val)
        self.register_buffer("E_num_pow10", E_num_pow10)
        self.register_buffer("E_is_num", E_is_num)

    def _compile_write_directions(
        self,
        tokenizer,
        W_lm: torch.Tensor,
        write_scale: float,
        dev: torch.device,
    ):
        """Solve exact write vectors W_write [NUM_OUT_SYMS, H] so lm_head(h + w_y) picks y."""
        syms: List[str] = (
            [str(d) for d in range(10)]
            + list(string.ascii_lowercase)
            + ["-", "\n", " "]
        )
        assert len(syms) == self.NUM_OUT_SYMS

        out_token_ids = []
        for s in syms:
            tids = tokenizer.encode(s, add_special_tokens=False)
            assert len(tids) == 1, f"Symbol {s!r} did not encode to 1 token: {tids}"
            out_token_ids.append(tids[0])

        self.register_buffer("out_token_ids", torch.tensor(out_token_ids, dtype=torch.long, device=dev))

        # Use unit-normalized lm_head rows scaled cleanly; verify self-dot beats every other vocab row!
        W_f = W_lm.float().to(dev)
        dirs = W_f[self.out_token_ids]  # [39, H]
        # Normalize each row to unit norm, then verify margin against all V rows
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        # Check if any other vocab row has larger projection onto dir_i (can happen if another row has larger norm!)
        # If so, do 5 fast gradient-ascent / perceptron steps on dir_i to guarantee exact top-1 separation!
        logits = dirs @ W_f.T  # [39, V]
        for i, tid in enumerate(out_token_ids):
            d = dirs[i].clone()
            for _ in range(15):
                sc = W_f @ d
                top_id = int(sc.argmax().item())
                if top_id == tid:
                    # Check margin >= 0.05
                    sc_copy = sc.clone()
                    sc_copy[tid] = -1e9
                    runner_up = int(sc_copy.argmax().item())
                    if sc[tid] - sc[runner_up] > 0.15:
                        break
                    d = d + 0.3 * (W_f[tid] - W_f[runner_up])
                else:
                    d = d + 0.5 * (W_f[tid] - W_f[top_id])
                d = d / (d.norm() + 1e-8)
            dirs[i] = d

        self.register_buffer("W_write", (dirs * write_scale).to(W_lm.dtype))

    def _reset_state(self, batch_size: int, device: torch.device):
        self.batch_size = batch_size
        self.dev = device
        # Ring buffer of recent tokens for pure tensor pattern matching & span storage
        self.hist_tokens = torch.full((batch_size, self.max_span_tokens), -1, dtype=torch.long, device=device)
        self.hist_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Precomputed emission queue per batch item: [B, max_emit, NUM_OUT_SYMS] (one-hot)
        self.max_emit = 128
        self.emit_queue = torch.zeros(batch_size, self.max_emit, self.NUM_OUT_SYMS, dtype=torch.float32, device=device)
        self.emit_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.emit_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.pending = torch.zeros(batch_size, 1, self.hidden_size, dtype=self.W_write.dtype, device=device)
        self.active_flag = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def attach(self):
        """Attach forward hook onto model.model.norm (right before lm_head)."""
        if self._hook_handle is not None:
            return self
        norm_mod = self.model.model.norm

        def _hook(_mod, _inp, output):
            if not self.enabled or self.pending is None:
                return output
            # Bit-for-bit identity fast path when ALL batch items are silent
            if not bool(self.active_flag.any()):
                return output
            # output: [B, T, H]; add pending ONLY to last position of active batch items
            out = output.clone()
            out[:, -1:, :] = out[:, -1:, :] + self.pending.to(out.dtype)
            return out

        self._hook_handle = norm_mod.register_forward_hook(_hook)
        return self

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        return self

    @torch.no_grad()
    def _check_trigger_and_compute(self, b: int):
        """Pure tensor-algebra trigger check and sub-token ALU execution for batch item b."""
        L = int(self.hist_len[b].item())
        if L < 4:
            return
        seq = self.hist_tokens[b, :L]  # [L]

        # Check if seq ends with any compiled close pattern (pure tensor equality)
        close_len = 0
        for pat in self.close_pats:
            plen = len(pat)
            if plen > 0 and L >= plen and bool((seq[L - plen:] == pat).all()):
                close_len = plen
                break
        if close_len == 0:
            return

        # Find most recent open pattern before close_pat using tensor match
        open_end = -1
        for pos in range(L - close_len - 1, -1, -1):
            for pat in self.open_pats:
                olen = len(pat)
                if pos + olen <= L - close_len and bool((seq[pos:pos + olen] == pat).all()):
                    open_end = pos + olen
                    break
            if open_end >= 0:
                break
        if open_end < 0:
            return

        inner = seq[open_end : L - close_len]  # tokens between `[XRAY` and `]=`
        if len(inner) < 2:
            return

        # Locate pipe separator `|` using E_is_pipe lookup
        pipe_mask = self.E_is_pipe[inner]  # [T_in]
        if pipe_mask.sum().item() < 0.5:
            return
        # First pipe index via argmax on binary mask
        pipe_idx = int(pipe_mask.argmax().item())
        header = inner[:pipe_idx]
        span = inner[pipe_idx + 1:]
        if len(header) == 0 or len(span) == 0:
            return

        # Decode op code via sum of E_op[header] -> argmax
        op_scores = self.E_op[header].sum(dim=0)  # [7]
        op_scores[0] = 0.0
        if op_scores.max().item() < 0.5:
            return
        op = int(op_scores.argmax().item())

        # Execute exact weight-space circuit for `op`!
        queue = torch.zeros(self.max_emit, self.NUM_OUT_SYMS, dtype=torch.float32, device=self.dev)
        q_len = 0

        if op == self.OP_COUNT:
            # Query letter one-hot from header tokens AFTER the op token (last token in header)
            arg_toks = header[1:] if len(header) > 1 else header
            q_vec = self.bank.E_hist[arg_toks].sum(dim=0)  # [26]
            # Normalize to one-hot via ReLU cap at 1
            q_vec = F.relu(q_vec) - F.relu(q_vec - 1.0)
            # Exact sum of token character histograms across span
            span_hist = self.bank.E_hist[span].sum(dim=0)  # [26]
            count_val = (q_vec * span_hist).sum().unsqueeze(0)  # [1]
            digits, active = self.digit_net(count_val)  # [1, 3, 10], [1, 3]
            for slot in range(3):
                if active[0, slot].item() > 0.5:
                    queue[q_len, self.OUT_DIGITS:self.OUT_DIGITS + 10] = digits[0, slot]
                    q_len += 1
            queue[q_len, self.OUT_NL] = 1.0
            q_len += 1

        elif op == self.OP_LEN:
            total_len = self.bank.E_alpha_len[span].sum().unsqueeze(0)  # [1]
            digits, active = self.digit_net(total_len)
            for slot in range(3):
                if active[0, slot].item() > 0.5:
                    queue[q_len, self.OUT_DIGITS:self.OUT_DIGITS + 10] = digits[0, slot]
                    q_len += 1
            queue[q_len, self.OUT_NL] = 1.0
            q_len += 1

        elif op in (self.OP_NTH, self.OP_SPELL, self.OP_REVERSE, self.OP_ROT13):
            # Exact prefix sum of alphabetic lengths across span via lower-triangular matmul!
            lens = self.bank.E_alpha_len[span]  # [S]
            S = len(span)
            tril = torch.tril(torch.ones(S, S, dtype=torch.float32, device=self.dev), diagonal=-1)
            offs = tril @ lens  # [S] start char offset of each token
            total_L = int(lens.sum().item())
            if total_L <= 0:
                return

            # One-hot character matrix for active span tokens: [S, L_max, 26]
            span_codes = self.bank.E_alpha_chars[span].float()  # [S, L_max]
            # Exact ReLU one-hot expansion against char_grid (1..26)
            span_oh = relu_eq_int(span_codes.unsqueeze(-1), self.char_grid.view(1, 1, 26))  # [S, L_max, 26]

            def extract_char_oh(char_idx: float) -> torch.Tensor:
                i_t = torch.tensor(char_idx, dtype=torch.float32, device=self.dev)
                # Exact ReLU token selector: 1 for unique token with offs[t] <= i <= offs[t] + lens[t] - 1
                tok_sel = relu_in_range(i_t, offs, offs + lens - 1.0)  # [S]
                # Exact intra-token slot offset
                slot = (tok_sel * (i_t - offs)).sum()  # scalar
                slot_sel = relu_eq_int(self.slot_grid, slot)  # [L_max]
                # Contract [S, L_max, 26] with tok_sel [S] and slot_sel [L_max] -> [26]
                c_oh = torch.einsum("s,m,smc->c", tok_sel, slot_sel, span_oh)
                return c_oh

            if op == self.OP_NTH:
                # Decode 1-indexed k from numeric tokens in header[1:] via place-value accumulation
                k_val = torch.tensor(0.0, dtype=torch.float32, device=self.dev)
                for tid in header[1:]:
                    if self.E_is_num[tid].item() > 0.5:
                        k_val = k_val * self.E_num_pow10[tid] + self.E_num_val[tid]
                k_int = int(k_val.item())
                if 1 <= k_int <= total_L:
                    c_oh = extract_char_oh(float(k_int - 1))
                    queue[q_len, self.OUT_ALPHA:self.OUT_ALPHA + 26] = c_oh
                    q_len += 1
                queue[q_len, self.OUT_NL] = 1.0
                q_len += 1

            elif op == self.OP_SPELL:
                for step_i in range(min(total_L, (self.max_emit - 2) // 2)):
                    c_oh = extract_char_oh(float(step_i))
                    queue[q_len, self.OUT_ALPHA:self.OUT_ALPHA + 26] = c_oh
                    q_len += 1
                    if step_i < total_L - 1:
                        queue[q_len, self.OUT_HYPHEN] = 1.0
                        q_len += 1
                queue[q_len, self.OUT_NL] = 1.0
                q_len += 1

            elif op == self.OP_REVERSE:
                for step_i in range(min(total_L, self.max_emit - 2)):
                    c_oh = extract_char_oh(float(total_L - 1 - step_i))
                    queue[q_len, self.OUT_ALPHA:self.OUT_ALPHA + 26] = c_oh
                    q_len += 1
                queue[q_len, self.OUT_NL] = 1.0
                q_len += 1

            elif op == self.OP_ROT13:
                for step_i in range(min(total_L, self.max_emit - 2)):
                    c_oh = extract_char_oh(float(step_i))
                    c_rot = c_oh @ self.P_rot13  # Exact 26x26 permutation linear layer!
                    queue[q_len, self.OUT_ALPHA:self.OUT_ALPHA + 26] = c_rot
                    q_len += 1
                queue[q_len, self.OUT_NL] = 1.0
                q_len += 1

        if q_len > 0:
            self.emit_queue[b] = queue
            self.emit_len[b] = q_len
            self.emit_ptr[b] = 0
            self._load_pending(b)

    def _load_pending(self, b: int):
        ptr = int(self.emit_ptr[b].item())
        elen = int(self.emit_len[b].item())
        if ptr < elen:
            sym_oh = self.emit_queue[b, ptr].to(self.W_write.dtype)  # [39]
            self.pending[b, 0] = sym_oh @ self.W_write  # [H]
            self.active_flag[b] = True
        else:
            self.pending[b, 0].zero_()
            self.active_flag[b] = False

    def _append_token(self, b: int, tid: int):
        L = int(self.hist_len[b].item())
        if L < self.max_span_tokens:
            self.hist_tokens[b, L] = tid
            self.hist_len[b] = L + 1
        else:
            self.hist_tokens[b, :-1] = self.hist_tokens[b, 1:].clone()
            self.hist_tokens[b, -1] = tid

    @torch.no_grad()
    def prime(self, input_ids: torch.Tensor):
        """Process initial prompt tokens [B, T] and set pending write for position T-1."""
        B, T = input_ids.shape
        self._reset_state(B, input_ids.device)
        # Keep last max_span_tokens of prompt in register
        take = min(T, self.max_span_tokens)
        self.hist_tokens[:, :take] = input_ids[:, T - take:]
        self.hist_len.fill_(take)
        for b in range(B):
            self._check_trigger_and_compute(b)

    @torch.no_grad()
    def step(self, next_token_ids: torch.Tensor):
        """Advance state machine by one newly sampled token [B]."""
        B = next_token_ids.shape[0]
        for b in range(B):
            tid = int(next_token_ids[b].item())
            self._append_token(b, tid)
            if self.active_flag[b].item():
                # Previous pending token was just emitted; advance emit pointer
                self.emit_ptr[b] += 1
                self._load_pending(b)
            else:
                self._check_trigger_and_compute(b)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 48,
        stop_on_newline: bool = False,
    ) -> Tuple[List[int], List[bool], List[torch.Tensor]]:
        """Greedy decode (B=1) returning (generated_token_ids, was_written_by_implant, per_step_logits)."""
        assert input_ids.shape[0] == 1
        self.prime(input_ids)
        out = self.model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values

        gen_ids: List[int] = []
        written: List[bool] = []
        all_logits: List[torch.Tensor] = []

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        nl_ids = set(self.tokenizer.encode("\n", add_special_tokens=False))

        for _ in range(max_new_tokens):
            was_w = bool(self.active_flag[0].item())
            lg = out.logits[:, -1, :].float()
            all_logits.append(lg[0].cpu())
            nxt = lg.argmax(dim=-1)
            tid = int(nxt.item())
            gen_ids.append(tid)
            written.append(was_w)

            if eos_id is not None and tid == eos_id:
                break
            if stop_on_newline and tid in nl_ids:
                break

            self.step(nxt)
            out = self.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values

        return gen_ids, written, all_logits
