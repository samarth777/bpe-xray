"""Act III (Breakthrough): Mid-Layer Weight-Space Visual Cortex ("BPE-V1").

Unlike naive Layer-0 embedding injection (where the host doesn't know what question it's
answering yet), BPE-V1 bolts lightweight causal cross-attention taps onto mid-to-late
transformer layers (e.g. layers 14, 20, 24 of Qwen3-0.6B):

  - Query  Q comes from the host's hidden state h_l ("what sub-token property do I need?")
  - Key    K combines host context h_l + frozen retina stream R_t ("which tokens & features?")
  - Value  V comes STRICTLY from the frozen tokenizer-compiled retina stream R_t!
  - Output W_o is EXACT ZERO-INITIALIZED -> bit-for-bit identical at step 0!

Because V reads STRICTLY from R_t (compiled from tokenizer.json: exact per-token histograms,
positional chars, cross-token boundary double-letter dot product, and causal prefix sums),
the implant cannot cheat by memorizing BPE token semantics — and works even when 100% of
the host LLM's weights remain COMPLETELY FROZEN!
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_tables import CharWeightBank, RETINA_SLICES


class RetinaFeatureStream(nn.Module):
    """Pure tensor-op builder of the frozen sub-token visual stream R_t [B, T, D_v1].

    Features per token position t (414 dims, ALL exact & deterministic from tokenizer):
      - [0:358]   f_raw[t]          : exact F_retina[x_t] (hist, first/last, pos_first6, pos_last4, scalars)
      - [358:385] prev_last[t]      : E_last[x_{t-1}] (27 dims: last letter of previous token)
      - [385:386] cross_double[t]   : exact dot product (E_last[x_{t-1}][1:] · E_first[x_t][1:]) -> 1.0 iff cross-BPE double!
      - [386:412] cum_hist[t] / 8.0 : exact causal prefix sum of E_hist[0:26] across sequence
      - [412:414] cum_len[t] / 16.0 : exact causal prefix sum of alpha_len and raw_len
    """

    DIM = 414

    def __init__(self, bank: CharWeightBank):
        super().__init__()
        self.register_buffer("F_retina", bank.F_retina.float().clone())  # [V, 358]
        self.feature_override_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    def compute_stream(
        self,
        input_ids: torch.Tensor,  # [B, T]
        prev_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = input_ids.shape
        f = self.F_retina[input_ids]  # [B, T, 358]
        if self.feature_override_fn is not None:
            f = self.feature_override_fn(input_ids, f)

        hist = f[:, :, 0:26]                  # [B, T, 26]
        first_a = f[:, :, 26:53]              # [B, T, 27]
        last_a = f[:, :, 53:80]               # [B, T, 27]
        raw_len = f[:, :, 350:351] * 10.0     # [B, T, 1]
        alpha_len = f[:, :, 351:352] * 10.0   # [B, T, 1]

        if prev_state is None:
            zero_last = torch.zeros(B, 1, 27, device=f.device, dtype=f.dtype)
            prev_last = torch.cat([zero_last, last_a[:, :-1, :]], dim=1) if T > 1 else zero_last
            cum_hist = torch.cumsum(hist, dim=1)
            cum_alpha = torch.cumsum(alpha_len, dim=1)
            cum_raw = torch.cumsum(raw_len, dim=1)
        else:
            # Incremental step (T == 1 during KV-cached decoding)
            p_last = prev_state["last_a"]          # [B, 1, 27]
            prev_last = torch.cat([p_last, last_a[:, :-1, :]], dim=1) if T > 1 else p_last
            cum_hist = prev_state["cum_hist"] + torch.cumsum(hist, dim=1)
            cum_alpha = prev_state["cum_alpha"] + torch.cumsum(alpha_len, dim=1)
            cum_raw = prev_state["cum_raw"] + torch.cumsum(raw_len, dim=1)

        # Exact cross-token boundary double-letter detector: last letter of t-1 == first letter of t!
        cross_double = (prev_last[:, :, 1:] * first_a[:, :, 1:]).sum(dim=-1, keepdim=True)  # [B, T, 1]

        R = torch.cat(
            [
                f,
                prev_last,
                cross_double,
                cum_hist / 8.0,
                cum_alpha / 16.0,
                cum_raw / 16.0,
            ],
            dim=-1,
        )  # [B, T, 414]

        new_state = {
            "last_a": last_a[:, -1:, :].detach(),
            "cum_hist": cum_hist[:, -1:, :].detach(),
            "cum_alpha": cum_alpha[:, -1:, :].detach(),
            "cum_raw": cum_raw[:, -1:, :].detach(),
        }
        return R, new_state


class V1CrossAttentionTap(nn.Module):
    """Single mid-layer causal cross-attention tap from host h_l onto Retina stream R.

    Values V depend STRICTLY on R (pure sub-token features).
    Final output projection `o_proj` is EXACT ZERO-INITIALIZED -> bit-identical at step 0.
    """

    def __init__(
        self,
        hidden_size: int,
        retina_dim: int = RetinaFeatureStream.DIM,
        attn_dim: int = 512,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.h_norm = nn.RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, attn_dim, bias=False)
        self.k_h_proj = nn.Linear(hidden_size, attn_dim, bias=False)
        self.k_r_proj = nn.Linear(retina_dim, attn_dim, bias=False)
        # V reads STRICTLY from the frozen BPE Retina stream R (zero host semantic leakage!)
        self.v_r_proj = nn.Linear(retina_dim, attn_dim, bias=False)

        # Readout MLP combining attended visual signal with query context (gating what to extract)
        self.out_fc1 = nn.Linear(attn_dim, mlp_dim, bias=False)
        self.out_gate = nn.Linear(hidden_size, mlp_dim, bias=False)
        self.o_proj = nn.Linear(mlp_dim, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)

        # Zero-init o_proj -> EXACT 0.0 perturbation at initialization!
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        h: torch.Tensor,                          # [B, T_q, H]
        R: torch.Tensor,                          # [B, T_q, D_r]
        scale: float = 1.0,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        pad_mask_q: Optional[torch.Tensor] = None,  # [B, T_q] True = valid token
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if scale == 0.0:
            return torch.zeros_like(h), kv_cache

        B, T_q, _ = h.shape
        h_f = self.h_norm(h.float())
        R_f = R.float()

        q = self.q_proj(h_f).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)        # [B, nh, T_q, hd]
        k_new = (self.k_h_proj(h_f) + self.k_r_proj(R_f)).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_r_proj(R_f).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and "k" in kv_cache:
            k = torch.cat([kv_cache["k"], k_new], dim=2)
            v = torch.cat([kv_cache["v"], v_new], dim=2)
            valid_k = (
                torch.cat([kv_cache["valid_k"], pad_mask_q], dim=1)
                if (pad_mask_q is not None and kv_cache.get("valid_k") is not None)
                else None
            )
            new_cache = {"k": k.detach(), "v": v.detach(), "valid_k": valid_k.detach() if valid_k is not None else None}
        else:
            k, v = k_new, v_new
            valid_k = pad_mask_q
            new_cache = (
                {"k": k.detach(), "v": v.detach(), "valid_k": valid_k.detach() if valid_k is not None else None}
                if kv_cache is not None
                else None
            )

        T_k = k.shape[2]
        # Build exact causal + padding mask for SDPA
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T_q, T_k]
        if T_q > 1:
            # Causal mask where query i attends to keys <= (T_k - T_q + i)
            offset = T_k - T_q
            q_idx = torch.arange(T_q, device=h.device).unsqueeze(1) + offset
            k_idx = torch.arange(T_k, device=h.device).unsqueeze(0)
            causal_mask = k_idx <= q_idx  # [T_q, T_k]
            attn_scores = attn_scores.masked_fill(~causal_mask.view(1, 1, T_q, T_k), -1e4)
        if valid_k is not None:
            attn_scores = attn_scores.masked_fill(~valid_k.view(B, 1, 1, T_k), -1e4)

        attn_w = F.softmax(attn_scores, dim=-1)
        ctx = torch.matmul(attn_w, v)  # [B, nh, T_q, hd]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T_q, self.attn_dim)

        # Gated readout: h_l selects which component of the attended visual vector to write
        gated = F.silu(self.out_gate(h_f)) * self.out_fc1(ctx)
        delta = self.o_proj(self.drop(gated)) * float(scale)
        return delta.to(h.dtype), new_cache


class VisualCortexLM(nn.Module):
    """Bolts BPE-V1 Visual Cortex taps onto mid-to-late layers of any CausalLM.

    Works seamlessly with BOTH training (`model(input_ids=..., labels=...)`) AND
    KV-cached generation (`model.generate(...)`), including left-padded batching!
    """

    def __init__(
        self,
        model: nn.Module,
        bank: CharWeightBank,
        tap_layers: Sequence[int] = (14, 20, 24),
        attn_dim: int = 512,
        num_heads: int = 8,
        mlp_dim: int = 512,
        pad_token_id: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.tap_layers = tuple(tap_layers)
        self.pad_token_id = pad_token_id
        hidden_size = model.config.hidden_size
        dev = next(model.parameters()).device

        self.stream = RetinaFeatureStream(bank).to(dev)
        self.taps = nn.ModuleDict({
            str(l): V1CrossAttentionTap(
                hidden_size=hidden_size,
                retina_dim=RetinaFeatureStream.DIM,
                attn_dim=attn_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            ).to(dev)
            for l in self.tap_layers
        })

        # Runtime state for forward hooks
        self.retina_scale: float = 1.0
        self._cur_R: Optional[torch.Tensor] = None
        self._cur_pad_mask: Optional[torch.Tensor] = None
        self._caching: bool = False
        self._stream_state: Optional[Dict[str, torch.Tensor]] = None
        self._layer_kv: Dict[str, Dict[str, torch.Tensor]] = {}
        self._handles: List = []
        self.attach()

    def set_retina_scale(self, scale: float):
        self.retina_scale = float(scale)

    def set_feature_override(self, fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]):
        self.stream.feature_override_fn = fn

    def reset_cache(self):
        self._stream_state = None
        self._layer_kv = {str(l): {} for l in self.tap_layers}

    def attach(self):
        if self._handles:
            return self
        embed = self.model.get_input_embeddings()

        def _embed_hook(_mod, inp, _out):
            input_ids = inp[0]  # [B, T]
            if self._caching:
                R, self._stream_state = self.stream.compute_stream(input_ids, prev_state=self._stream_state)
            else:
                R, _ = self.stream.compute_stream(input_ids, prev_state=None)
            self._cur_R = R
            if self.pad_token_id is not None:
                self._cur_pad_mask = (input_ids != self.pad_token_id)
            else:
                self._cur_pad_mask = None

        self._handles.append(embed.register_forward_hook(_embed_hook))

        layers = self.model.model.layers
        for l_idx in self.tap_layers:
            key = str(l_idx)
            tap = self.taps[key]

            def _make_layer_hook(k=key, t_mod=tap):
                def _layer_hook(_mod, _inp, out):
                    if self.retina_scale == 0.0 or self._cur_R is None:
                        return out
                    is_tuple = isinstance(out, tuple)
                    h = out[0] if is_tuple else out
                    kv = self._layer_kv[k] if self._caching else None
                    delta, new_kv = t_mod(
                        h,
                        self._cur_R,
                        scale=self.retina_scale,
                        kv_cache=kv,
                        pad_mask_q=self._cur_pad_mask,
                    )
                    if self._caching and new_kv is not None:
                        self._layer_kv[k] = new_kv
                    h_new = h + delta
                    if is_tuple:
                        return (h_new,) + out[1:]
                    return h_new
                return _layer_hook

            self._handles.append(layers[l_idx].register_forward_hook(_make_layer_hook()))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    def forward(self, *args, **kwargs):
        self._caching = False
        self._stream_state = None
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self._caching = True
        self.reset_cache()
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            self._caching = False
            self._stream_state = None
            self._layer_kv = {}
