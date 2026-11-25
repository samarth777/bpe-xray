"""Act III Training & Evaluation: Continuous BPE Retina Sensory Organ vs Plain LoRA.

Strictly controlled apples-to-apples comparison on Qwen3-0.6B:
  - `base`        : Frozen base model (0 training)
  - `plain_lora`  : Standard LoRA (r=18, NO retina -> strictly >= trainable params than retina_lora!)
  - `optic_only`  : Completely FROZEN LLM + trainable Layer-0 Optic Nerve ONLY (~0.36M params!)
  - `retina_lora` : Frozen BPE Retina + trainable Optic Nerve + LoRA (r=16)

Also runs automatically on trained `retina_lora` / `optic_only`:
  1. Held-out Real English Words (7 tasks)
  2. Fresh Unseen Nonce / Pseudowords (7 tasks — impossible to memorize!)
  3. Cross-Token Boundary Double-Letter stress test
  4. Held-out OOD Tasks (Multi-word letter count + Anagrams — NEVER seen during training!)
  5. Causal Blindfold Test (`retina_scale = 0.0` at inference)
  6. Causal Optical Illusion Sweep (perturb ONLY `F_retina['r']` on 'strawberry' & 20 words!)
  7. General English Prose Perplexity preservation check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import (
    CharExample,
    build_retina_datasets,
    generate_nonce_words,
    has_cross_token_double,
    has_double_letter,
    make_example,
)
from char_tables import CharWeightBank
from xray_retina import RetinaAugmentedLM, apply_lora


PROSE_EVAL_TEXTS = [
    "The mountain trail wound steadily upward through dense groves of Douglas fir and cedar, opening occasionally onto rocky overlooks where the valley below lay shrouded in pale morning mist.",
    "In 1905 Albert Einstein published four groundbreaking papers that reshaped modern physics, addressing the photoelectric effect, Brownian motion, special relativity, and mass-energy equivalence.",
    "A well-designed compiler separates lexical analysis, parsing, intermediate representation optimization, and target code generation into clean modular stages.",
    "During autumn migrations, flocks of sandhill cranes gather along shallow river sandbars at dusk, creating a chorus of resonant calls that echoes across the plains for miles.",
    "To bake sourdough bread with an open crumb, bakers rely on long cold fermentation, high hydration dough, gentle stretch-and-fold handling, and steam during the first twenty minutes.",
    "Marine biologists studying hydrothermal vents discovered entire ecosystems thriving without sunlight, powered instead by chemosynthetic bacteria oxidizing hydrogen sulfide.",
]


def format_pair(ex: CharExample) -> Tuple[str, str]:
    """Return (prompt_text, target_text) cleanly aligned for causal LM loss on answer only."""
    return ex.prompt, " " + ex.answer + "\n"


def collate_batch(batch: List[CharExample], tok, max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_seqs = []
    label_seqs = []
    for ex in batch:
        p_txt, a_txt = format_pair(ex)
        p_ids = tok.encode(p_txt, add_special_tokens=False)
        a_ids = tok.encode(a_txt, add_special_tokens=False)
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        input_seqs.append(ids)
        label_seqs.append(labels)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = max(len(s) for s in input_seqs)
    B = len(batch)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, T), dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    for i, (ids, lbs) in enumerate(zip(input_seqs, label_seqs)):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, :L] = 1
        labels[i, :L] = torch.tensor(lbs, dtype=torch.long)
    return input_ids, attn_mask, labels


@torch.no_grad()
def evaluate_prose_ppl(model, tok, device: torch.device) -> float:
    model.eval()
    total_nll = 0.0
    total_toks = 0
    for txt in PROSE_EVAL_TEXTS:
        ids = torch.tensor([tok.encode(txt, add_special_tokens=False)], device=device)
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += float(out.loss.item()) * n
        total_toks += n
    return math.exp(total_nll / max(1, total_toks))


@torch.no_grad()
def eval_batched_greedy(
    model,
    tok,
    examples: List[CharExample],
    device: torch.device,
    batch_size: int = 32,
    max_new_tokens: int = 22,
    fewshot_prefix: str = "",
) -> Tuple[float, Dict[str, float], List[Dict]]:
    """Fast left-padded batched greedy decoding for exact match accuracy."""
    model.eval()
    orig_padding_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    nl_id = tok.encode("\n", add_special_tokens=False)[-1]
    correct_by_task = defaultdict(int)
    total_by_task = defaultdict(int)
    rows = []

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [fewshot_prefix + ex.prompt for ex in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        prompt_len = enc["input_ids"].shape[1]
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=[nl_id, tok.eos_token_id] if tok.eos_token_id is not None else nl_id,
            pad_token_id=tok.pad_token_id,
        )
        gen_only = out_ids[:, prompt_len:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        for ex, raw in zip(chunk, texts):
            pred = raw.strip().split("\n")[0].strip().rstrip(".,;:").lower()
            ok = (pred == ex.answer.lower())
            correct_by_task[ex.task] += int(ok)
            total_by_task[ex.task] += 1
            rows.append({
                "task": ex.task,
                "word": ex.word,
                "gold": ex.answer,
                "pred": pred,
                "ok": ok,
            })

    tok.padding_side = orig_padding_side
    per_task = {
        t: round(100.0 * correct_by_task[t] / max(1, total_by_task[t]), 2)
        for t in sorted(total_by_task.keys())
    }
    overall = round(100.0 * sum(correct_by_task.values()) / max(1, sum(total_by_task.values())), 2)
    return overall, per_task, rows


@torch.no_grad()
def run_optical_illusion_suite(
    wrapped: RetinaAugmentedLM,
    tok,
    words: List[str],
    device: torch.device,
) -> Dict:
    """Surgical counterfactual causal proof:
    Keep input_ids 100% IDENTICAL, perturb ONLY F_retina['letter'] on the word's last token
    by delta in [-2, -1, 0, +1, +2, +3], and check if the model's predicted count shifts by delta!
    """
    wrapped.eval()
    nl_id = tok.encode("\n", add_special_tokens=False)[-1]

    # Helper: find exact token positions belonging to the last `'{word}'` span in prompt
    def find_word_token_span(prompt_str: str, w: str) -> Tuple[List[int], List[int]]:
        marker = f"'{w}'"
        idx = prompt_str.rfind(marker)
        assert idx >= 0
        pre_ids = tok.encode(prompt_str[: idx + 1], add_special_tokens=False)
        full_ids = tok.encode(prompt_str, add_special_tokens=False)
        end_pre_ids = tok.encode(prompt_str[: idx + 1 + len(w)], add_special_tokens=False)
        # Span of tokens covering `w`
        span_pos = list(range(len(pre_ids), min(len(end_pre_ids), len(full_ids))))
        if not span_pos:
            span_pos = [max(0, len(pre_ids) - 1)]
        return full_ids, span_pos

    # 1. Headline demo: "strawberry" with delta in [-2, -1, 0, +1, +2, +3]
    strawberry_sweep = []
    prompt_sb = "How many times does the letter 'r' appear in 'strawberry'?\nAnswer:"
    full_ids_sb, span_sb = find_word_token_span(prompt_sb, "strawberry")
    ids_sb = torch.tensor([full_ids_sb], device=device)
    sb_tokens = [tok.decode([i], clean_up_tokenization_spaces=False) for i in full_ids_sb]
    r_idx = ord("r") - ord("a")

    for delta in [-2, -1, 0, 1, 2, 3]:
        def _override(input_ids, f_raw, _span=span_sb, _d=delta):
            f_mod = f_raw.clone()
            if f_mod.shape[1] > max(_span):
                if _d >= 0:
                    # Add all positive delta onto the last token of the word
                    f_mod[:, _span[-1], r_idx] = f_mod[:, _span[-1], r_idx] + float(_d)
                else:
                    # Subtract cleanly across tokens in _span that currently have r > 0
                    rem = float(-_d)
                    for pos in reversed(_span):
                        cur = float(f_mod[0, pos, r_idx].item())
                        take = min(cur, rem)
                        f_mod[:, pos, r_idx] = cur - take
                        rem -= take
                        if rem <= 1e-6:
                            break
            return f_mod

        wrapped.set_feature_override(_override)
        out = wrapped.generate(
            input_ids=ids_sb,
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=[nl_id, tok.eos_token_id] if tok.eos_token_id is not None else nl_id,
            pad_token_id=tok.pad_token_id,
        )
        pred = tok.decode(out[0, ids_sb.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()
        expected = str(max(0, 3 + delta))
        strawberry_sweep.append({
            "delta": delta,
            "true_count": 3,
            "illusion_target": expected,
            "model_pred": pred,
            "followed_illusion": (pred == expected),
        })
    wrapped.set_feature_override(None)

    # 2. Systematic 25-word optical illusion fidelity test (+1, +2 shifts)
    rng = random.Random(2026)
    n_total = 0
    n_followed = 0
    for w in words[:25]:
        present = [c for c in set(w) if w.count(c) >= 1]
        ch = rng.choice(present)
        c_idx = ord(ch) - ord("a")
        true_cnt = w.count(ch)
        prompt = f"How many times does the letter '{ch}' appear in '{w}'?\nAnswer:"
        full_ids, span_w = find_word_token_span(prompt, w)
        ids = torch.tensor([full_ids], device=device)
        tpos = span_w[-1]

        for delta in [1, 2]:
            def _ov(input_ids, f_raw, _pos=tpos, _ci=c_idx, _d=delta):
                f_mod = f_raw.clone()
                if f_mod.shape[1] > _pos >= 0:
                    f_mod[:, _pos, _ci] = f_mod[:, _pos, _ci] + float(_d)
                return f_mod

            wrapped.set_feature_override(_ov)
            out = wrapped.generate(
                input_ids=ids,
                max_new_tokens=6,
                do_sample=False,
                eos_token_id=[nl_id, tok.eos_token_id] if tok.eos_token_id is not None else nl_id,
                pad_token_id=tok.pad_token_id,
            )
            pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()
            target = str(true_cnt + delta)
            n_total += 1
            if pred == target:
                n_followed += 1
    wrapped.set_feature_override(None)

    return {
        "strawberry_tokens": sb_tokens,
        "perturbed_token_span": span_sb,
        "perturbed_token_strs": [sb_tokens[p] for p in span_sb],
        "strawberry_sweep": strawberry_sweep,
        "systematic_25w_fidelity": round(100.0 * n_followed / max(1, n_total), 2),
        "systematic_25w_n": n_total,
    }


def build_cross_token_double_suite(tok, seed: int = 42, n: int = 80) -> List[CharExample]:
    """Build balanced yes/no double-letter test set where ALL 'yes' cases straddle BPE boundaries!"""
    cand = generate_nonce_words(2500, seed=seed + 700, min_len=8, max_len=15, random_mix=0.5)
    pos_cross = []
    neg_no_double = []
    for w in cand:
        if has_double_letter(w):
            if has_cross_token_double(w, tok):
                # Extra crisp: check that NO single token has an internal double (pure cross-boundary!)
                ids = tok.encode(" " + w, add_special_tokens=False)
                pieces = [tok.decode([i], clean_up_tokenization_spaces=False) for i in ids]
                if not any(has_double_letter("".join(c for c in p if c.isalpha())) for p in pieces):
                    pos_cross.append(w)
        else:
            neg_no_double.append(w)
        if len(pos_cross) >= n // 2 and len(neg_no_double) >= n // 2:
            break

    rng = random.Random(seed + 701)
    exs = []
    for w in pos_cross[: n // 2]:
        exs.append(make_example("double", w, rng, split_tag="cross_double"))
    for w in neg_no_double[: n // 2]:
        exs.append(make_example("double", w, rng, split_tag="cross_double"))
    rng.shuffle(exs)
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--arm", required=True, choices=["base", "plain_lora", "optic_only", "retina_lora"])
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--optic_lr", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"=== BPE Retina Experiment | arm={args.arm} | model={args.model} | n_train={args.n_train} | seed={args.seed} ===")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)

    vocab_size = base_model.get_input_embeddings().weight.shape[0]
    bank = CharWeightBank.from_tokenizer(tok, vocab_size=vocab_size, verbose=True)

    # Measure base prose perplexity before any training
    base_ppl = evaluate_prose_ppl(base_model, tok, device)
    print(f"[PPL] Untouched base prose perplexity: {base_ppl:.3f}")

    wrapped: Optional[RetinaAugmentedLM] = None
    trainable_params_info = {}

    if args.arm == "base":
        eval_model = base_model
        trainable_params_info = {"lora": 0, "optic": 0, "total": 0}
    elif args.arm == "plain_lora":
        # Give plain_lora r=18 so it has STRICTLY MORE trainable params than retina_lora (r=16 + optic)!
        n_lora = apply_lora(base_model, r=args.lora_r + 2, alpha=float((args.lora_r + 2) * 2))
        eval_model = base_model
        trainable_params_info = {"lora": n_lora, "optic": 0, "total": n_lora}
    elif args.arm == "optic_only":
        wrapped = RetinaAugmentedLM(base_model, bank=bank, bottleneck_dim=args.bottleneck)
        n_optic = sum(p.numel() for p in wrapped.optic.parameters() if p.requires_grad)
        eval_model = wrapped
        trainable_params_info = {"lora": 0, "optic": n_optic, "total": n_optic}
    elif args.arm == "retina_lora":
        n_lora = apply_lora(base_model, r=args.lora_r, alpha=float(args.lora_r * 2))
        wrapped = RetinaAugmentedLM(base_model, bank=bank, bottleneck_dim=args.bottleneck)
        n_optic = sum(p.numel() for p in wrapped.optic.parameters() if p.requires_grad)
        eval_model = wrapped
        trainable_params_info = {"lora": n_lora, "optic": n_optic, "total": n_lora + n_optic}

    print(f"[Params] Trainable params: {trainable_params_info}")

    # Verify step-0 bit-identity if retina is attached!
    if wrapped is not None:
        test_ids = torch.tensor([tok.encode(PROSE_EVAL_TEXTS[0], add_special_tokens=False)], device=device)
        with torch.no_grad():
            wrapped.set_retina_scale(0.0)
            lg0 = wrapped(input_ids=test_ids).logits
            wrapped.set_retina_scale(1.0)
            lg1 = wrapped(input_ids=test_ids).logits
        print(f"[Verify] Step-0 zero-init bit-identical: {bool(torch.equal(lg0, lg1))}")

    data = build_retina_datasets(n_train=args.n_train, n_test_per_task=args.n_test, seed=args.seed)
    train_exs = data["train"]

    # Train!
    train_seconds = 0.0
    final_loss = None
    if args.arm != "base":
        param_groups = []
        lora_params = [p for n, p in base_model.named_parameters() if p.requires_grad]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr, "weight_decay": 0.01})
        if wrapped is not None:
            optic_params = [p for p in wrapped.optic.parameters() if p.requires_grad]
            param_groups.append({"params": optic_params, "lr": args.optic_lr, "weight_decay": 0.01})

        opt = torch.optim.AdamW(param_groups)
        # Scale epochs up slightly for tiny n_train so small-N sweeps converge cleanly
        eff_epochs = args.epochs
        if args.n_train <= 256:
            eff_epochs = max(args.epochs, 12)
        elif args.n_train <= 1024:
            eff_epochs = max(args.epochs, 6)

        total_steps = eff_epochs * math.ceil(len(train_exs) / args.batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1e-5)

        t0 = time.time()
        rng_shuf = random.Random(args.seed + 10)
        step = 0
        for ep in range(eff_epochs):
            eval_model.train()
            order = list(train_exs)
            rng_shuf.shuffle(order)
            ep_loss = 0.0
            ep_batches = 0
            for start in range(0, len(order), args.batch_size):
                chunk = order[start : start + args.batch_size]
                input_ids, attn_mask, labels = collate_batch(chunk, tok)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)

                out = eval_model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], 1.0
                )
                opt.step()
                sched.step()
                ep_loss += float(loss.item())
                ep_batches += 1
                step += 1
            final_loss = ep_loss / max(1, ep_batches)
            print(f"  [Epoch {ep+1}/{eff_epochs}] step={step}/{total_steps} loss={final_loss:.4f}")
        train_seconds = time.time() - t0
        print(f"[Train] Done in {train_seconds:.1f}s")

    # Evaluate across all suites!
    # For `base`, give 2-shot generic format prefix? Actually evaluating zero-shot AND/OR 1-shot helps base format cleanly;
    # let's evaluate strictly zero-shot for trained models, and for `base` evaluate BOTH 0-shot and 2-shot so base gets every fair chance!
    fewshot_for_base = ""
    if args.arm == "base":
        fewshot_for_base = (
            "How many times does the letter 'a' appear in 'banana'?\nAnswer: 3\n\n"
            "How many letters are in the word 'banana'?\nAnswer: 6\n\n"
            "Reverse the letters of 'planet'.\nAnswer: tenalp\n\n"
        )

    cross_double_exs = build_cross_token_double_suite(tok, seed=args.seed, n=80)
    print(f"[Cross-Double Suite] Built {len(cross_double_exs)} pure cross-BPE-boundary double-letter tests")

    real_ov, real_pt, _ = eval_batched_greedy(eval_model, tok, data["test_real"], device, fewshot_prefix=fewshot_for_base)
    nonce_ov, nonce_pt, _ = eval_batched_greedy(eval_model, tok, data["test_nonce"], device, fewshot_prefix=fewshot_for_base)
    multi_ov, _, _ = eval_batched_greedy(eval_model, tok, data["test_multiword"], device, fewshot_prefix=fewshot_for_base)
    anag_ov, _, _ = eval_batched_greedy(eval_model, tok, data["test_anagram"], device, fewshot_prefix=fewshot_for_base)
    cross_ov, _, _ = eval_batched_greedy(eval_model, tok, cross_double_exs, device, fewshot_prefix=fewshot_for_base)
    post_ppl = evaluate_prose_ppl(eval_model, tok, device)

    print(f"\n=== RESULTS ({args.arm}, n_train={args.n_train}) ===")
    print(f"Held-out Real Words (7 tasks) : {real_ov:5.1f}% | {real_pt}")
    print(f"Unseen Nonce Words  (7 tasks) : {nonce_ov:5.1f}% | {nonce_pt}")
    print(f"Pure Cross-BPE Double Letters : {cross_ov:5.1f}%")
    print(f"OOD Multi-Word Count          : {multi_ov:5.1f}%")
    print(f"OOD Anagram Check             : {anag_ov:5.1f}%")
    print(f"Normal English Prose PPL      : {post_ppl:.3f} (base={base_ppl:.3f})")

    causal_results = {}
    if wrapped is not None:
        # Blindfold Test!
        wrapped.set_retina_scale(0.0)
        bf_real, bf_real_pt, _ = eval_batched_greedy(eval_model, tok, data["test_real"], device)
        bf_nonce, bf_nonce_pt, _ = eval_batched_greedy(eval_model, tok, data["test_nonce"], device)
        bf_cross, _, _ = eval_batched_greedy(eval_model, tok, cross_double_exs, device)
        wrapped.set_retina_scale(1.0)
        print(f"\n=== CAUSAL BLINDFOLD TEST (retina_scale=0.0) ===")
        print(f"Real Words  : {real_ov:5.1f}% -> {bf_real:5.1f}% (drop = {bf_real - real_ov:+.1f} pts)")
        print(f"Nonce Words : {nonce_ov:5.1f}% -> {bf_nonce:5.1f}% (drop = {bf_nonce - nonce_ov:+.1f} pts)")
        print(f"Cross-Double: {cross_ov:5.1f}% -> {bf_cross:5.1f}% (drop = {bf_cross - cross_ov:+.1f} pts)")

        # Optical Illusion Sweep!
        illusion = run_optical_illusion_suite(wrapped, tok, data["test_words"], device)
        print(f"\n=== CAUSAL OPTICAL ILLUSION TEST (unchanged text tokens, perturbed retina) ===")
        for row in illusion["strawberry_sweep"]:
            print(
                f"  strawberry delta={row['delta']:+d} -> illusion_target={row['illusion_target']} | "
                f"model_pred={row['model_pred']!r} | followed={row['followed_illusion']}"
            )
        print(f"Systematic 25-word illusion fidelity: {illusion['systematic_25w_fidelity']:.1f}% (n={illusion['systematic_25w_n']})")

        causal_results = {
            "blindfold_real": bf_real,
            "blindfold_real_per_task": bf_real_pt,
            "blindfold_nonce": bf_nonce,
            "blindfold_nonce_per_task": bf_nonce_pt,
            "blindfold_cross_double": bf_cross,
            "optical_illusion": illusion,
        }

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    out_path = args.out or f"results/retina_{slug}_{args.arm}_n{args.n_train}_s{args.seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "model": args.model,
        "arm": args.arm,
        "n_train": args.n_train,
        "seed": args.seed,
        "trainable_params": trainable_params_info,
        "train_seconds": round(train_seconds, 2),
        "final_loss": final_loss,
        "prose_ppl_base": round(base_ppl, 4),
        "prose_ppl_post": round(post_ppl, 4),
        "test_real_overall": real_ov,
        "test_real_per_task": real_pt,
        "test_nonce_overall": nonce_ov,
        "test_nonce_per_task": nonce_pt,
        "test_cross_double": cross_ov,
        "test_ood_multiword": multi_ov,
        "test_ood_anagram": anag_ov,
        "causal": causal_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path}")

    if args.save_ckpt and args.arm != "base":
        ckpt_path = out_path.replace(".json", ".pt")
        state = {}
        if wrapped is not None:
            state["optic"] = wrapped.optic.state_dict()
        lora_sd = {k: v.cpu() for k, v in base_model.state_dict().items() if "lora_" in k}
        if lora_sd:
            state["lora"] = lora_sd
        torch.save(state, ckpt_path)
        print(f"Saved lightweight implant checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
