"""Evaluate saved Plain LoRA (r=18) checkpoint on Compound BPE Chunks + Adversarial BPE Splits
so every table & chart is 100% apples-to-apples with BPE-V2!"""
from __future__ import annotations

import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import get_word_splits, make_example
from eval_adversarial_bpe import eval_model_on_splits, random_valid_bpe_split
from train_retina import eval_batched_greedy
from train_v1 import build_compound_bpe_pseudowords
from xray_retina import apply_lora


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_name = "Qwen/Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    apply_lora(model, r=18, alpha=36.0)
    sd = torch.load("results/retina_Qwen_Qwen3-0.6B-Base_plain_lora_n4096_s42.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd["lora"], strict=False)

    # 1. Compound BPE pseudowords (exact same seed as train_v1/v2!)
    compound_words = build_compound_bpe_pseudowords(tok, n=120, seed=42 + 800)
    rng_cmp = random.Random(42 + 801)
    tasks_7 = ("count", "length", "nth", "reverse", "double", "delete", "replace")
    compound_exs = [
        make_example(tasks_7[i % len(tasks_7)], compound_words[i % len(compound_words)], rng_cmp, "compound")
        for i in range(len(tasks_7) * 40)
    ]
    comp_ov, comp_pt, _ = eval_batched_greedy(model, tok, compound_exs, device)
    print(f"[Plain LoRA r=18] Compound BPE Chunks (7 tasks): {comp_ov:5.1f}% | {comp_pt}")

    # 2. Adversarial BPE splits (exact same seed as train_v1/v2!)
    _, test_words = get_word_splits(seed=42)
    rng_adv = random.Random(42 + 505)
    adv_tasks = ["count", "length", "nth", "reverse"]
    adv_exs, canon_ids, adv_ids = [], [], []
    for t in adv_tasks:
        for i in range(40):
            w = test_words[i % len(test_words)]
            adv_exs.append(make_example(t, w, rng_adv, "real"))
            canon_ids.append(tok.encode(w, add_special_tokens=False))
            adv_ids.append(random_valid_bpe_split(w, tok, rng_adv))
    adv_canon, _ = eval_model_on_splits(model, tok, adv_exs, canon_ids, device)
    adv_split, _ = eval_model_on_splits(model, tok, adv_exs, adv_ids, device)
    print(f"[Plain LoRA r=18] Canonical BPE: {adv_canon:5.1f}% | Adversarial Random BPE Splits: {adv_split:5.1f}%")

    out = {
        "test_compound_overall": comp_ov,
        "test_compound_per_task": comp_pt,
        "adversarial_bpe": {"canonical": adv_canon, "adversarial": adv_split},
    }
    with open("results/plain_lora_n4096_extra.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/plain_lora_n4096_extra.json")


if __name__ == "__main__":
    main()
