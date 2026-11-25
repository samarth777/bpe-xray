"""Adversarial BPE Re-Tokenization Invariance Experiment!

Take held-out words and feed the EXACT SAME string under:
  1. Canonical BPE tokenization
  2. Random adversarial valid BPE splits (e.g. ["s", "traw", "ber", "ry"] or ["st", "raw", "b", "erry"])
     that decode byte-for-byte to the exact same text!

Compares:
  - Plain LoRA (trained at n=4096)
  - BPE Retina + LoRA (trained at n=4096)
  - Exact BPE X-Ray ALU (zero training!)
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import get_word_splits, make_example
from char_tables import CharWeightBank
from xray_alu import XRayALUImplant
from xray_retina import RetinaAugmentedLM, apply_lora


def random_valid_bpe_split(word: str, tok, rng: random.Random, min_chunks: int = 2) -> List[int]:
    """Produce a random valid tokenization of `word` whose decode equals `word` exactly."""
    L = len(word)
    if L <= 2:
        return tok.encode(word, add_special_tokens=False)
    # Try up to 20 random split patterns into 2..4 non-empty pieces
    canon = tok.encode(word, add_special_tokens=False)
    for _ in range(25):
        n_cuts = rng.randint(min_chunks - 1, min(4, L - 1))
        cuts = sorted(rng.sample(range(1, L), k=n_cuts))
        bounds = [0] + cuts + [L]
        ids: List[int] = []
        ok = True
        for a, b in zip(bounds, bounds[1:]):
            piece = word[a:b]
            pids = tok.encode(piece, add_special_tokens=False)
            if tok.decode(pids, clean_up_tokenization_spaces=False) != piece:
                ok = False
                break
            ids.extend(pids)
        if ok and ids != canon and tok.decode(ids, clean_up_tokenization_spaces=False) == word:
            return ids
    # Fallback: character-by-character tokenization
    ids = []
    for ch in word:
        ids.extend(tok.encode(ch, add_special_tokens=False))
    return ids


def encode_prompt_with_word_ids(prompt: str, word: str, word_ids: List[int], tok) -> List[int]:
    """Splice custom `word_ids` into `prompt` where `'{word}'` appears."""
    marker = f"'{word}'"
    idx = prompt.rfind(marker)
    assert idx >= 0, f"{marker} not in {prompt}"
    prefix = prompt[: idx + 1]  # includes opening quote
    suffix = prompt[idx + 1 + len(word) :]  # includes closing quote onward
    return (
        tok.encode(prefix, add_special_tokens=False)
        + word_ids
        + tok.encode(suffix, add_special_tokens=False)
    )


@torch.no_grad()
def eval_model_on_splits(model, tok, examples, word_id_lists, device) -> Tuple[float, List[Dict]]:
    model.eval()
    nl_id = tok.encode("\n", add_special_tokens=False)[-1]
    correct = 0
    rows = []
    for ex, wids in zip(examples, word_id_lists):
        ids = encode_prompt_with_word_ids(ex.prompt, ex.word, wids, tok)
        inp = torch.tensor([ids], device=device)
        out = model.generate(
            input_ids=inp,
            max_new_tokens=18,
            do_sample=False,
            eos_token_id=[nl_id, tok.eos_token_id] if tok.eos_token_id is not None else nl_id,
            pad_token_id=tok.pad_token_id,
        )
        pred = tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".,;:").lower()
        ok = (pred == ex.answer.lower())
        correct += int(ok)
        rows.append({"task": ex.task, "word": ex.word, "gold": ex.answer, "pred": pred, "ok": ok})
    return round(100.0 * correct / max(1, len(examples)), 2), rows


@torch.no_grad()
def eval_alu_on_splits(alu: XRayALUImplant, tok, examples, word_id_lists, device) -> float:
    correct = 0
    for ex, wids in zip(examples, word_id_lists):
        assert ex.alu_call is not None
        # Splice custom word_ids right after `| ` inside `[XRAY ... | <word>]=`
        prefix_s, suffix_s = ex.alu_call.split(ex.word, 1)
        ids = (
            tok.encode(prefix_s, add_special_tokens=False)
            + wids
            + tok.encode(suffix_s, add_special_tokens=False)
        )
        inp = torch.tensor([ids], device=device)
        gen_ids, _, _ = alu.generate_greedy(inp, max_new_tokens=32, stop_on_newline=True)
        pred = tok.decode(gen_ids).strip().split("\n")[0].strip().lower()
        correct += int(pred == ex.answer.lower())
    return round(100.0 * correct / max(1, len(examples)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--retina_ckpt", required=True)
    ap.add_argument("--plain_ckpt", default=None)
    ap.add_argument("--n_per_task", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/adversarial_bpe.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    _, test_words = get_word_splits(seed=args.seed)
    rng = random.Random(args.seed + 505)
    tasks = ["count", "length", "nth", "reverse"]
    exs = []
    canon_ids = []
    adv_ids = []
    for task in tasks:
        for i in range(args.n_per_task):
            w = test_words[i % len(test_words)]
            exs.append(make_example(task, w, rng, split_tag="real"))
            canon_ids.append(tok.encode(w, add_special_tokens=False))
            adv_ids.append(random_valid_bpe_split(w, tok, rng))

    # 1. Exact ALU on canonical vs adversarial BPE splits
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    bank = CharWeightBank.from_tokenizer(tok, vocab_size=base_model.get_input_embeddings().weight.shape[0])
    alu = XRayALUImplant(base_model, tok, bank=bank, device=device).attach()
    alu_canon = eval_alu_on_splits(alu, tok, exs, canon_ids, device)
    alu_adv = eval_alu_on_splits(alu, tok, exs, adv_ids, device)
    alu.detach()
    print(f"[X-Ray ALU]   Canonical BPE: {alu_canon:5.1f}% | Adversarial BPE Splits: {alu_adv:5.1f}%")

    # 2. Load Retina + LoRA checkpoint
    apply_lora(base_model, r=16, alpha=32.0)
    wrapped = RetinaAugmentedLM(base_model, bank=bank, bottleneck_dim=256)
    sd = torch.load(args.retina_ckpt, map_location=device, weights_only=True)
    wrapped.optic.load_state_dict(sd["optic"])
    if "lora" in sd:
        base_model.load_state_dict(sd["lora"], strict=False)
    ret_canon, _ = eval_model_on_splits(wrapped, tok, exs, canon_ids, device)
    ret_adv, _ = eval_model_on_splits(wrapped, tok, exs, adv_ids, device)
    print(f"[Retina+LoRA] Canonical BPE: {ret_canon:5.1f}% | Adversarial BPE Splits: {ret_adv:5.1f}%")

    # Blindfold on adversarial!
    wrapped.set_retina_scale(0.0)
    bf_canon, _ = eval_model_on_splits(wrapped, tok, exs, canon_ids, device)
    bf_adv, _ = eval_model_on_splits(wrapped, tok, exs, adv_ids, device)
    wrapped.set_retina_scale(1.0)
    print(f"[Blindfolded] Canonical BPE: {bf_canon:5.1f}% | Adversarial BPE Splits: {bf_adv:5.1f}%")

    # 3. Optional: Plain LoRA checkpoint on canonical vs adversarial BPE splits
    plain_res = None
    if args.plain_ckpt is not None:
        plain_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
        for p in plain_model.parameters():
            p.requires_grad_(False)
        apply_lora(plain_model, r=18, alpha=36.0)
        psd = torch.load(args.plain_ckpt, map_location=device, weights_only=True)
        plain_model.load_state_dict(psd["lora"], strict=False)
        pl_canon, _ = eval_model_on_splits(plain_model, tok, exs, canon_ids, device)
        pl_adv, _ = eval_model_on_splits(plain_model, tok, exs, adv_ids, device)
        print(f"[Plain LoRA]  Canonical BPE: {pl_canon:5.1f}% | Adversarial BPE Splits: {pl_adv:5.1f}%")
        plain_res = {"canonical": pl_canon, "adversarial": pl_adv}

    res = {
        "n_examples": len(exs),
        "tasks": tasks,
        "alu": {"canonical": alu_canon, "adversarial": alu_adv},
        "retina_lora": {"canonical": ret_canon, "adversarial": ret_adv},
        "retina_blindfolded": {"canonical": bf_canon, "adversarial": bf_adv},
        "plain_lora": plain_res,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
