"""Train & evaluate BPE-V1 Mid-Layer Visual Cortex on a 100% FROZEN Qwen3-0.6B (and optional +LoRA)!

Compares directly against Plain LoRA (11.35M params) and Layer-0 Retina:
  - `v1_frozen` : 100% FROZEN Qwen3-0.6B + Mid-Layer BPE-V1 Visual Cortex (~5.5M params, 0 host weights changed!)
  - `v1_lora`   : BPE-V1 Visual Cortex + lightweight LoRA (r=8, total params STILL < Plain LoRA's 11.35M!)

Also evaluates:
  - Held-out Real English Words (7 tasks)
  - Unseen Nonce / Pseudowords (7 tasks)
  - Compound Multi-Chunk Pseudowords (TRUE BPE blindness: glued 4-7 char BPE chunks!)
  - Pure Cross-BPE-Boundary Double Letters
  - OOD Multi-Word Counting & Anagrams (never seen in training!)
  - Causal Blindfold Test (`retina_scale = 0.0` -> bit-identical untouched host!)
  - Causal Optical Illusion Sweep on 'strawberry' + 25 held-out words!
  - Adversarial BPE Re-Tokenization Invariance (built right in!)
  - Prose Perplexity preservation check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import (
    CharExample,
    build_retina_datasets,
    get_word_splits,
    make_example,
)
from char_tables import CharWeightBank
from eval_adversarial_bpe import eval_model_on_splits, random_valid_bpe_split
from train_retina import (
    PROSE_EVAL_TEXTS,
    build_cross_token_double_suite,
    collate_batch,
    eval_batched_greedy,
    evaluate_prose_ppl,
    run_optical_illusion_suite,
)
from visual_cortex import VisualCortexLM
from xray_retina import apply_lora


def build_compound_bpe_pseudowords(tok, n: int = 120, seed: int = 31415) -> List[str]:
    """Build compound pseudowords glued from real 4-7 char BPE chunks (TRUE BPE blindness!).
    Unlike random gibberish (which BPE splits into 1-char tokens), these compress into just
    2-3 long BPE chunks — impossible to memorize AND impossible to read without sub-token eyes!
    """
    chunks = [
        "straw", "berry", "graph", "photo", "micro", "chrono", "trans", "inter",
        "super", "hyper", "ultra", "cyber", "electro", "thermo", "neuro", "hydro",
        "astro", "crypto", "macro", "proto", "pseudo", "multi", "counter", "under",
        "over", "bright", "shadow", "silver", "golden", "crystal", "marble", "copper",
        "velvet", "scarlet", "crimson", "emerald", "sapphire", "amber", "violet",
        "storm", "frost", "flame", "spark", "river", "mountain", "forest", "meadow",
        "harbor", "castle", "garden", "bridge", "tower", "compass", "lantern", "anchor",
        "mirror", "window", "ribbon", "button", "pocket", "basket", "kettle", "hammer",
        "planet", "rocket", "comet", "nebula", "quasar", "photon", "proton", "matrix",
        "vector", "tensor", "cipher", "beacon", "voyage", "summit", "canyon", "glacier",
        "rhythm", "melody", "chorus", "sonnet", "ballad", "fable", "legend", "chronicle",
    ]
    rng = random.Random(seed)
    out = []
    seen = set()
    while len(out) < n:
        a, b = rng.sample(chunks, 2)
        w = a + b
        if w in seen:
            continue
        # Verify BPE compresses it into <= 3 tokens while having >= 10 chars!
        tids = tok.encode(w, add_special_tokens=False)
        if len(w) >= 10 and len(tids) <= 3:
            seen.add(w)
            out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--arm", default="v1_frozen", choices=["v1_frozen", "v1_lora"])
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--tap_layers", default="14,20,24")
    ap.add_argument("--attn_dim", type=int, default=512)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--prose_kl_weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tap_layers = tuple(int(x) for x in args.tap_layers.split(","))

    print(f"=== BPE-V1 Visual Cortex | arm={args.arm} | model={args.model} | taps={tap_layers} | n_train={args.n_train} ===")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)

    vocab_size = base_model.get_input_embeddings().weight.shape[0]
    bank = CharWeightBank.from_tokenizer(tok, vocab_size=vocab_size, verbose=True)

    base_ppl = evaluate_prose_ppl(base_model, tok, device)
    print(f"[PPL] Untouched base prose perplexity: {base_ppl:.3f}")

    n_lora = 0
    if args.arm == "v1_lora":
        n_lora = apply_lora(base_model, r=args.lora_r, alpha=float(args.lora_r * 2))

    wrapped = VisualCortexLM(
        base_model,
        bank=bank,
        tap_layers=tap_layers,
        attn_dim=args.attn_dim,
        num_heads=8,
        mlp_dim=args.mlp_dim,
        pad_token_id=tok.pad_token_id,
    )
    n_v1 = sum(p.numel() for p in wrapped.taps.parameters() if p.requires_grad)
    trainable_params_info = {"lora": n_lora, "v1_cortex": n_v1, "total": n_lora + n_v1}
    print(f"[Params] Trainable params: {trainable_params_info}")

    # Verify step-0 bit-for-bit identity!
    test_ids = torch.tensor([tok.encode(PROSE_EVAL_TEXTS[0], add_special_tokens=False)], device=device)
    with torch.no_grad():
        wrapped.set_retina_scale(0.0)
        lg0 = wrapped(input_ids=test_ids).logits
        wrapped.set_retina_scale(1.0)
        lg1 = wrapped(input_ids=test_ids).logits
    print(f"[Verify] Step-0 zero-init bit-identical: {bool(torch.equal(lg0, lg1))}")

    data = build_retina_datasets(n_train=args.n_train, n_test_per_task=args.n_test, seed=args.seed)
    train_exs = data["train"]

    # Pre-encode normal prose anchor sentences for tiny KL preservation (only touches V1 outputs on prose!)
    prose_anchor_ids = [
        torch.tensor([tok.encode(t, add_special_tokens=False)], device=device)
        for t in PROSE_EVAL_TEXTS[:3]
    ]
    prose_anchor_logits = []
    if args.prose_kl_weight > 0:
        with torch.no_grad():
            wrapped.set_retina_scale(0.0)
            for pids in prose_anchor_ids:
                prose_anchor_logits.append(wrapped(input_ids=pids).logits.detach().float())
            wrapped.set_retina_scale(1.0)

    param_groups = [{"params": list(wrapped.taps.parameters()), "lr": args.lr, "weight_decay": 0.01}]
    lora_params = [p for n, p in base_model.named_parameters() if p.requires_grad]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": 2e-4, "weight_decay": 0.01})

    opt = torch.optim.AdamW(param_groups)
    eff_epochs = args.epochs
    if args.n_train <= 256:
        eff_epochs = max(args.epochs, 14)
    elif args.n_train <= 1024:
        eff_epochs = max(args.epochs, 7)

    total_steps = eff_epochs * math.ceil(len(train_exs) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1e-5)

    t0 = time.time()
    rng_shuf = random.Random(args.seed + 10)
    step = 0
    final_loss = 0.0
    for ep in range(eff_epochs):
        wrapped.train()
        order = list(train_exs)
        rng_shuf.shuffle(order)
        ep_loss = 0.0
        ep_b = 0
        for start in range(0, len(order), args.batch_size):
            chunk = order[start : start + args.batch_size]
            input_ids, attn_mask, labels = collate_batch(chunk, tok)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            out = wrapped(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss

            # Every 8 steps, tiny KL anchor on prose so V1 stays quiet on normal English!
            if args.prose_kl_weight > 0 and (step % 8 == 0):
                pidx = (step // 8) % len(prose_anchor_ids)
                cur_lg = wrapped(input_ids=prose_anchor_ids[pidx]).logits.float()
                ref_lg = prose_anchor_logits[pidx]
                kl = F.kl_div(
                    F.log_softmax(cur_lg, dim=-1),
                    F.softmax(ref_lg, dim=-1),
                    reduction="batchmean",
                )
                loss = loss + args.prose_kl_weight * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0
            )
            opt.step()
            sched.step()
            ep_loss += float(out.loss.item())
            ep_b += 1
            step += 1
        final_loss = ep_loss / max(1, ep_b)
        print(f"  [Epoch {ep+1}/{eff_epochs}] step={step}/{total_steps} task_loss={final_loss:.4f}")

    train_seconds = time.time() - t0
    print(f"[Train] Done in {train_seconds:.1f}s")

    # Build held-out evaluation suites
    cross_double_exs = build_cross_token_double_suite(tok, seed=args.seed, n=80)
    compound_words = build_compound_bpe_pseudowords(tok, n=120, seed=args.seed + 800)
    rng_cmp = random.Random(args.seed + 801)
    tasks_7 = ("count", "length", "nth", "reverse", "double", "delete", "replace")
    compound_exs = [
        make_example(tasks_7[i % len(tasks_7)], compound_words[i % len(compound_words)], rng_cmp, "compound")
        for i in range(len(tasks_7) * 40)
    ]

    # Evaluate on held-out prose texts (indices 3..5 NEVER seen in KL anchor!)
    wrapped.eval()
    heldout_ppl_base = 0.0
    wrapped.set_retina_scale(0.0)
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for txt in PROSE_EVAL_TEXTS[3:]:
            ids = torch.tensor([tok.encode(txt, add_special_tokens=False)], device=device)
            n = ids.shape[1] - 1
            total_nll += float(wrapped(input_ids=ids, labels=ids).loss.item()) * n
            total_toks += n
    heldout_ppl_base = math.exp(total_nll / max(1, total_toks))

    wrapped.set_retina_scale(1.0)
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for txt in PROSE_EVAL_TEXTS[3:]:
            ids = torch.tensor([tok.encode(txt, add_special_tokens=False)], device=device)
            n = ids.shape[1] - 1
            total_nll += float(wrapped(input_ids=ids, labels=ids).loss.item()) * n
            total_toks += n
    heldout_ppl_post = math.exp(total_nll / max(1, total_toks))

    real_ov, real_pt, _ = eval_batched_greedy(wrapped, tok, data["test_real"], device)
    nonce_ov, nonce_pt, _ = eval_batched_greedy(wrapped, tok, data["test_nonce"], device)
    comp_ov, comp_pt, _ = eval_batched_greedy(wrapped, tok, compound_exs, device)
    cross_ov, _, _ = eval_batched_greedy(wrapped, tok, cross_double_exs, device)
    multi_ov, _, _ = eval_batched_greedy(wrapped, tok, data["test_multiword"], device)
    anag_ov, _, _ = eval_batched_greedy(wrapped, tok, data["test_anagram"], device)

    print(f"\n=== RESULTS ({args.arm}, n_train={args.n_train}) ===")
    print(f"Held-out Real Words (7 tasks) : {real_ov:5.1f}% | {real_pt}")
    print(f"Unseen Nonce Words  (7 tasks) : {nonce_ov:5.1f}% | {nonce_pt}")
    print(f"Compound BPE Chunks (7 tasks) : {comp_ov:5.1f}% | {comp_pt}")
    print(f"Pure Cross-BPE Double Letters : {cross_ov:5.1f}%")
    print(f"OOD Multi-Word Count          : {multi_ov:5.1f}%")
    print(f"OOD Anagram Check             : {anag_ov:5.1f}%")
    print(f"Held-out Prose PPL            : {heldout_ppl_post:.3f} (untouched base={heldout_ppl_base:.3f})")

    # Causal Blindfold Test
    wrapped.set_retina_scale(0.0)
    bf_real, bf_real_pt, _ = eval_batched_greedy(wrapped, tok, data["test_real"], device)
    bf_nonce, bf_nonce_pt, _ = eval_batched_greedy(wrapped, tok, data["test_nonce"], device)
    bf_comp, _, _ = eval_batched_greedy(wrapped, tok, compound_exs, device)
    bf_cross, _, _ = eval_batched_greedy(wrapped, tok, cross_double_exs, device)
    wrapped.set_retina_scale(1.0)
    print(f"\n=== CAUSAL BLINDFOLD TEST (retina_scale=0.0) ===")
    print(f"Real Words     : {real_ov:5.1f}% -> {bf_real:5.1f}% (drop = {bf_real - real_ov:+.1f} pts)")
    print(f"Nonce Words    : {nonce_ov:5.1f}% -> {bf_nonce:5.1f}% (drop = {bf_nonce - nonce_ov:+.1f} pts)")
    print(f"Compound BPE   : {comp_ov:5.1f}% -> {bf_comp:5.1f}% (drop = {bf_comp - comp_ov:+.1f} pts)")
    print(f"Cross-Double   : {cross_ov:5.1f}% -> {bf_cross:5.1f}% (drop = {bf_cross - cross_ov:+.1f} pts)")

    # Causal Optical Illusion Sweep!
    illusion = run_optical_illusion_suite(wrapped, tok, data["test_words"], device)
    print(f"\n=== CAUSAL OPTICAL ILLUSION TEST (100% identical text tokens, perturbed retina) ===")
    for row in illusion["strawberry_sweep"]:
        print(
            f"  strawberry delta={row['delta']:+d} -> illusion_target={row['illusion_target']} | "
            f"model_pred={row['model_pred']!r} | followed={row['followed_illusion']}"
        )
    print(f"Systematic 25-word illusion fidelity: {illusion['systematic_25w_fidelity']:.1f}% (n={illusion['systematic_25w_n']})")

    # Adversarial BPE Re-Tokenization Invariance (4 core tasks, 40 each)
    rng_adv = random.Random(args.seed + 505)
    adv_tasks = ["count", "length", "nth", "reverse"]
    adv_exs, canon_ids, adv_ids = [], [], []
    for t in adv_tasks:
        for i in range(40):
            w = data["test_words"][i % len(data["test_words"])]
            adv_exs.append(make_example(t, w, rng_adv, "real"))
            canon_ids.append(tok.encode(w, add_special_tokens=False))
            adv_ids.append(random_valid_bpe_split(w, tok, rng_adv))
    adv_canon, _ = eval_model_on_splits(wrapped, tok, adv_exs, canon_ids, device)
    adv_split, _ = eval_model_on_splits(wrapped, tok, adv_exs, adv_ids, device)
    print(f"\n=== ADVERSARIAL BPE RE-TOKENIZATION INVARIANCE ===")
    print(f"Canonical BPE: {adv_canon:5.1f}% | Adversarial Random BPE Splits: {adv_split:5.1f}%")

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    out_path = args.out or f"results/v1_{slug}_{args.arm}_n{args.n_train}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "model": args.model,
        "arm": args.arm,
        "tap_layers": list(tap_layers),
        "n_train": args.n_train,
        "seed": args.seed,
        "trainable_params": trainable_params_info,
        "train_seconds": round(train_seconds, 2),
        "final_loss": final_loss,
        "prose_ppl_base_heldout": round(heldout_ppl_base, 4),
        "prose_ppl_post_heldout": round(heldout_ppl_post, 4),
        "test_real_overall": real_ov,
        "test_real_per_task": real_pt,
        "test_nonce_overall": nonce_ov,
        "test_nonce_per_task": nonce_pt,
        "test_compound_overall": comp_ov,
        "test_compound_per_task": comp_pt,
        "test_cross_double": cross_ov,
        "test_ood_multiword": multi_ov,
        "test_ood_anagram": anag_ov,
        "adversarial_bpe": {"canonical": adv_canon, "adversarial": adv_split},
        "causal": {
            "blindfold_real": bf_real,
            "blindfold_real_per_task": bf_real_pt,
            "blindfold_nonce": bf_nonce,
            "blindfold_nonce_per_task": bf_nonce_pt,
            "blindfold_compound": bf_comp,
            "blindfold_cross_double": bf_cross,
            "optical_illusion": illusion,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path}")

    if args.save_ckpt:
        ckpt_path = out_path.replace(".json", ".pt")
        state = {"taps": wrapped.taps.state_dict(), "tap_layers": list(tap_layers)}
        lora_sd = {k: v.cpu() for k, v in base_model.state_dict().items() if "lora_" in k}
        if lora_sd:
            state["lora"] = lora_sd
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
