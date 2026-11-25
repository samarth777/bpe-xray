"""Compile frozen weight-space character lookup tables directly from any HF tokenizer.

Once compiled at load time (~0.5-1.5s), the tokenizer is NEVER invoked at inference:
all sub-token character information lives strictly in frozen tensor buffers indexed
by token IDs.
"""
from __future__ import annotations

import string
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# Character alphabet for exact sub-token representation:
# 0 = PAD
# 1..26 = 'a'..'z'
# 27 = space (' ')
# 28 = other printable
CHAR_PAD = 0
CHAR_A = 1
CHAR_SPACE = 27
CHAR_OTHER = 28
NUM_CHAR_CLASSES = 29

ALPHA_CHARS = string.ascii_lowercase  # 26
VOWELS = set("aeiou")
DIGITS = set("0123456789")


def char_to_code(ch: str) -> int:
    c = ch.lower()
    if len(c) == 1 and "a" <= c <= "z":
        return CHAR_A + (ord(c) - ord("a"))
    if ch == " ":
        return CHAR_SPACE
    return CHAR_OTHER


def code_to_char(code: int) -> str:
    if CHAR_A <= code < CHAR_A + 26:
        return chr(ord("a") + (code - CHAR_A))
    if code == CHAR_SPACE:
        return " "
    return "?"


@dataclass
class RetinaSlices:
    """Exact index slices inside F_retina [V, D_retina] for interpretability & causal edits."""
    hist: slice          # [0:26]    exact lowercase a..z counts
    first_alpha: slice   # [26:53]   one-hot (0=none, 1..26=a..z) of first letter
    last_alpha: slice    # [53:80]   one-hot (0=none, 1..26=a..z) of last letter
    pos_first: slice     # [80:242]  first 6 chars (6 * 27 one-hots: 0=none, 1..26=a..z)
    pos_last: slice      # [242:350] last 4 chars reversed from end (4 * 27 one-hots)
    scalars: slice       # [350:358] length, alpha_len, vowels, consonants, lead_space, internal_double, is_upper, has_digit
    dim: int = 358


RETINA_SLICES = RetinaSlices(
    hist=slice(0, 26),
    first_alpha=slice(26, 53),
    last_alpha=slice(53, 80),
    pos_first=slice(80, 242),
    pos_last=slice(242, 350),
    scalars=slice(350, 358),
    dim=358,
)


class CharWeightBank(nn.Module):
    """Frozen weight-space sub-token lookup tables compiled from a BPE tokenizer.

    Buffers:
      E_hist:         [V, 26]         exact lowercase a..z count per token
      E_len:          [V]             exact stripped/raw char length (without leading BPE space for word ops)
      E_raw_len:      [V]             exact total decoded char length
      E_alpha_len:    [V]             exact alphabetic char count
      E_lead_space:   [V]             1.0 if token starts with space else 0.0
      E_chars:        [V, L_max]      long tensor of char codes (0..28) for exact ALU indexing (raw chars)
      E_alpha_chars:  [V, L_max]      long tensor of alphabetic char codes (1..26, 0=pad)
      F_retina:       [V, 358]        rich continuous sub-token feature vector for BPE Retina
    """

    def __init__(
        self,
        vocab_size: int,
        max_tok_chars: int = 24,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tok_chars = max_tok_chars
        self.register_buffer("E_hist", torch.zeros(vocab_size, 26, dtype=torch.float32))
        self.register_buffer("E_raw_len", torch.zeros(vocab_size, dtype=torch.float32))
        self.register_buffer("E_alpha_len", torch.zeros(vocab_size, dtype=torch.float32))
        self.register_buffer("E_lead_space", torch.zeros(vocab_size, dtype=torch.float32))
        self.register_buffer("E_chars", torch.zeros(vocab_size, max_tok_chars, dtype=torch.long))
        self.register_buffer("E_alpha_chars", torch.zeros(vocab_size, max_tok_chars, dtype=torch.long))
        self.register_buffer("F_retina", torch.zeros(vocab_size, RETINA_SLICES.dim, dtype=torch.float32))

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        vocab_size: Optional[int] = None,
        max_tok_chars: int = 24,
        verbose: bool = False,
    ) -> "CharWeightBank":
        t0 = time.time()
        tok_len = len(tokenizer)
        V = max(tok_len, vocab_size or tok_len)
        bank = cls(vocab_size=V, max_tok_chars=max_tok_chars)

        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        # Also include added special tokens if any
        if hasattr(tokenizer, "added_tokens_decoder"):
            for tid, info in tokenizer.added_tokens_decoder.items():
                if getattr(info, "special", False):
                    special_ids.add(tid)

        # Decode every token ID once at compile time
        ids_to_decode = [[i] for i in range(tok_len)]
        decoded = tokenizer.batch_decode(
            ids_to_decode,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        E_hist = bank.E_hist
        E_raw_len = bank.E_raw_len
        E_alpha_len = bank.E_alpha_len
        E_lead_space = bank.E_lead_space
        E_chars = bank.E_chars
        E_alpha_chars = bank.E_alpha_chars
        F_retina = bank.F_retina

        for tid, s in enumerate(decoded):
            if tid in special_ids or not s:
                continue
            # Skip invalid UTF-8 replacement-only byte fragments for char counting
            raw_len = min(len(s), max_tok_chars)
            E_raw_len[tid] = float(len(s))
            if s[0] == " ":
                E_lead_space[tid] = 1.0

            alphas = []
            vowels = 0
            consonants = 0
            has_digit = 0.0
            has_upper = 0
            has_lower = 0
            has_internal_double = 0.0
            prev_lower = ""

            for idx, ch in enumerate(s):
                if idx < max_tok_chars:
                    E_chars[tid, idx] = char_to_code(ch)
                cl = ch.lower()
                if len(cl) == 1 and "a" <= cl <= "z":
                    ci = ord(cl) - ord("a")
                    E_hist[tid, ci] += 1.0
                    alphas.append(ci + 1)  # 1..26
                    if cl in VOWELS:
                        vowels += 1
                    else:
                        consonants += 1
                    if ch.isupper():
                        has_upper += 1
                    else:
                        has_lower += 1
                    if cl == prev_lower:
                        has_internal_double = 1.0
                    prev_lower = cl
                else:
                    if ch in DIGITS:
                        has_digit = 1.0
                    prev_lower = ""

            n_alpha = len(alphas)
            E_alpha_len[tid] = float(n_alpha)
            for idx, ac in enumerate(alphas[:max_tok_chars]):
                E_alpha_chars[tid, idx] = ac

            # Populate F_retina[tid]
            F_retina[tid, 0:26] = E_hist[tid]
            if n_alpha > 0:
                F_retina[tid, 26 + alphas[0]] = 1.0
                F_retina[tid, 53 + alphas[-1]] = 1.0
            else:
                F_retina[tid, 26] = 1.0
                F_retina[tid, 53] = 1.0

            # First 6 alphabetic chars
            for pos in range(6):
                base = 80 + pos * 27
                code = alphas[pos] if pos < n_alpha else 0
                F_retina[tid, base + code] = 1.0

            # Last 4 alphabetic chars (pos=0 is last char, pos=1 is second-to-last, ...)
            for pos in range(4):
                base = 242 + pos * 27
                code = alphas[-(pos + 1)] if pos < n_alpha else 0
                F_retina[tid, base + code] = 1.0

            # 8 scalar features
            F_retina[tid, 350] = len(s) / 10.0
            F_retina[tid, 351] = n_alpha / 10.0
            F_retina[tid, 352] = vowels / 10.0
            F_retina[tid, 353] = consonants / 10.0
            F_retina[tid, 354] = 1.0 if s[0] == " " else 0.0
            F_retina[tid, 355] = has_internal_double
            F_retina[tid, 356] = 1.0 if (has_upper > 0 and has_lower == 0) else 0.0
            F_retina[tid, 357] = has_digit

        if verbose:
            dt = time.time() - t0
            mb_hist = E_hist.numel() * 2 / 1e6
            mb_ret = F_retina.numel() * 2 / 1e6
            print(
                f"[CharWeightBank] Compiled V={V:,} tokens in {dt:.2f}s "
                f"(E_hist bf16={mb_hist:.1f}MB, F_retina bf16={mb_ret:.1f}MB)"
            )
        return bank
