"""Act III: Continuous BPE Retina — Weight-Space Sub-Token Sensory Organ.

Injects frozen tokenizer-compiled sub-token character features directly into Layer 0's
residual stream via a zero-initialized Optic Nerve MLP:

    h_0 = embed_tokens(x) + retina_scale * W_optic(F_retina[x])

At initialization (step 0), W_optic's output projection is exact zeros, so the host LLM
is bit-for-bit unchanged. Supports surgical causal interventions at inference time:
  - Blindfold test: set `retina_scale = 0.0`
  - Optical illusion: pass `feature_override_fn(input_ids, f_raw) -> f_perturbed`
    to mutate ONLY the sub-token sensory signal while leaving text tokens untouched!
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_tables import CharWeightBank, RETINA_SLICES


class BPERetinaOpticNerve(nn.Module):
    """Frozen tokenizer feature table + zero-initialized, embed-scale-matched Optic Nerve.

    Design principles:
      - NO LayerNorm over F_retina -> preserves exact linear additivity of letter counts!
      - bias=False -> non-alphabetic/zero tokens receive near-zero perturbation.
      - Direct linear pathway (w_lin) + small SiLU MLP residual (fc1->fc2), both zero-init
        at output and scaled by `embed_scale` (RMS of host token embeddings, ~0.025) so
        early training never overwhelms pretrained Layer-0 embeddings!
    """

    def __init__(
        self,
        F_retina: torch.Tensor,  # [V, D_ret]
        hidden_size: int,
        bottleneck_dim: int = 256,
        embed_scale: float = 0.03,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("F_retina", F_retina.float().clone())
        self.register_buffer("embed_scale", torch.tensor(float(embed_scale), dtype=torch.float32))
        d_in = F_retina.shape[1]

        # Direct linear pathway (exact additive channel for counts & one-hots)
        self.w_lin = nn.Linear(d_in, hidden_size, bias=False)
        # Small nonlinear residual for intra-token positional interactions
        self.fc1 = nn.Linear(d_in, bottleneck_dim, bias=False)
        self.fc2 = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        # EXACT ZERO init on both output projections -> bit-for-bit identical at step 0!
        nn.init.zeros_(self.w_lin.weight)
        nn.init.zeros_(self.fc2.weight)

        # Runtime causal-intervention controls
        self.retina_scale: float = 1.0
        self.feature_override_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    def forward(self, input_ids: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        if self.retina_scale == 0.0:
            B, T = input_ids.shape
            return torch.zeros(B, T, self.w_lin.out_features, device=input_ids.device, dtype=target_dtype)

        f = self.F_retina[input_ids]  # [B, T, D_ret]
        if self.feature_override_fn is not None:
            f = self.feature_override_fn(input_ids, f)
        lin_out = self.w_lin(f)
        mlp_out = self.fc2(self.drop(F.silu(self.fc1(f))))
        delta = (lin_out + mlp_out) * (self.embed_scale * self.retina_scale)
        return delta.to(target_dtype)


class LoRALinear(nn.Module):
    """Minimal, dependency-free LoRA wrapper around nn.Linear."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / float(r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_f = self.drop(x.float())
        lora_out = F.linear(F.linear(x_f, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out.to(base_out.dtype)


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    target_names: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    layer_indices: Optional[Iterable[int]] = None,
) -> int:
    """Wrap matching Linear layers inside `model.model.layers` with LoRALinear. Returns trainable param count."""
    layers = model.model.layers
    allowed = set(layer_indices) if layer_indices is not None else set(range(len(layers)))
    added = 0
    for idx, layer in enumerate(layers):
        if idx not in allowed:
            continue
        for parent_name, parent_mod in list(layer.named_modules()):
            for child_name, child in list(parent_mod.named_children()):
                if child_name in target_names and isinstance(child, nn.Linear):
                    wrapped = LoRALinear(child, r=r, alpha=alpha)
                    setattr(parent_mod, child_name, wrapped)
                    added += wrapped.lora_A.numel() + wrapped.lora_B.numel()
    return added


class RetinaAugmentedLM(nn.Module):
    """Wraps a HuggingFace CausalLM with a Layer-0 BPE Retina Optic Nerve via forward hook.

    Keeps full compatibility with `model.generate(...)`, KV caching, and standard HF APIs!
    """

    def __init__(
        self,
        model: nn.Module,
        bank: CharWeightBank,
        bottleneck_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = model
        hidden_size = model.config.hidden_size
        embed_w = model.get_input_embeddings().weight.detach().float()
        embed_rms = float(embed_w.pow(2).mean().sqrt().item())
        dev = next(model.parameters()).device
        self.optic = BPERetinaOpticNerve(
            F_retina=bank.F_retina,
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            embed_scale=embed_rms,
            dropout=dropout,
        ).to(dev)
        self._hook_handle = None
        self.attach()

    def attach(self):
        if self._hook_handle is not None:
            return self
        embed = self.model.get_input_embeddings()

        def _embed_hook(_mod, inp, out):
            # inp[0] is input_ids [B, T]; out is embeddings [B, T, H]
            input_ids = inp[0]
            delta = self.optic(input_ids, target_dtype=out.dtype)
            return out + delta

        self._hook_handle = embed.register_forward_hook(_embed_hook)
        return self

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        return self

    def set_retina_scale(self, scale: float):
        self.optic.retina_scale = float(scale)

    def set_feature_override(self, fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]):
        self.optic.feature_override_fn = fn

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
