"""Act I & Act II Evaluation:
1. Verify exactness of compiled ReLU Digit Extractor (all integers 0..999).
2. Verify BIT-FOR-BIT IDENTICAL logits when X-Ray ALU is silent (normal prose + pre-trigger).
3. Evaluate Frozen Host Alone (2-shot) vs Frozen Host + Exact BPE X-Ray ALU (2-shot, ZERO training!)
   across 6 sub-token tasks (count, len, nth, spell, reverse, rot13) on:
     - Held-out real English words
     - Fresh unseen nonce/gibberish words
     - Multi-word phrases (count)
4. Supports arbitrary HF models (Qwen3-0.6B, Qwen3-1.7B, SmolLM2-1.7B-Instruct) to prove
   automatic cross-family compilation straight from tokenizer.json!
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import (
    CharExample,
    generate_nonce_words,
    get_word_splits,
    make_example,
    make_multiword_count_examples,
)
from char_tables import CharWeightBank
from xray_alu import ReLUDigitExtractor, XRayALUImplant


# 2-shot ICL templates teaching ONLY the compact `[XRAY ...]=` format (same spirit as WSM §1.2)
FEWSHOT_PLAIN: Dict[str, str] = {
    "count": (
        "Q: How many times does the letter 'a' appear in 'banana'?\nA: 3\n\n"
        "Q: How many times does the letter 'e' appear in 'elephant'?\nA: 2\n\n"
    ),
    "length": (
        "Q: How many letters are in the word 'banana'?\nA: 6\n\n"
        "Q: How many letters are in the word 'elephant'?\nA: 8\n\n"
    ),
    "nth": (
        "Q: What is the 3rd letter of 'banana'?\nA: n\n\n"
        "Q: What is the 5th letter of 'elephant'?\nA: h\n\n"
    ),
    "spell": (
        "Q: Spell 'cat' letter by letter separated by hyphens.\nA: c-a-t\n\n"
        "Q: Spell 'dog' letter by letter separated by hyphens.\nA: d-o-g\n\n"
    ),
    "reverse": (
        "Q: Reverse the letters of 'banana'.\nA: ananab\n\n"
        "Q: Reverse the letters of 'planet'.\nA: tenalp\n\n"
    ),
    "rot13": (
        "Q: Apply ROT13 to 'hello'.\nA: uryyb\n\n"
        "Q: Apply ROT13 to 'world'.\nA: jbeyq\n\n"
    ),
}

FEWSHOT_ALU: Dict[str, str] = {
    "count": (
        "Q: How many times does the letter 'a' appear in 'banana'?\nA: [XRAY count a | banana]=3\n\n"
        "Q: How many times does the letter 'e' appear in 'elephant'?\nA: [XRAY count e | elephant]=2\n\n"
    ),
    "length": (
        "Q: How many letters are in the word 'banana'?\nA: [XRAY len | banana]=6\n\n"
        "Q: How many letters are in the word 'elephant'?\nA: [XRAY len | elephant]=8\n\n"
    ),
    "nth": (
        "Q: What is the 3rd letter of 'banana'?\nA: [XRAY nth 3 | banana]=n\n\n"
        "Q: What is the 5th letter of 'elephant'?\nA: [XRAY nth 5 | elephant]=h\n\n"
    ),
    "spell": (
        "Q: Spell 'cat' letter by letter separated by hyphens.\nA: [XRAY spell | cat]=c-a-t\n\n"
        "Q: Spell 'dog' letter by letter separated by hyphens.\nA: [XRAY spell | dog]=d-o-g\n\n"
    ),
    "reverse": (
        "Q: Reverse the letters of 'banana'.\nA: [XRAY reverse | banana]=ananab\n\n"
        "Q: Reverse the letters of 'planet'.\nA: [XRAY reverse | planet]=tenalp\n\n"
    ),
    "rot13": (
        "Q: Apply ROT13 to 'hello'.\nA: [XRAY rot13 | hello]=uryyb\n\n"
        "Q: Apply ROT13 to 'world'.\nA: [XRAY rot13 | world]=jbeyq\n\n"
    ),
}


def verify_relu_digits(device: torch.device) -> bool:
    net = ReLUDigitExtractor().to(device)
    nums = torch.arange(0, 1000, device=device)
    digits, active = net(nums)
    for n in range(1000):
        s = ""
        for slot in range(3):
            if active[n, slot].item() > 0.5:
                s += str(int(digits[n, slot].argmax().item()))
        assert s == str(n), f"Digit mismatch for {n}: got {s!r}"
    return True


@torch.no_grad()
def verify_bit_identical(model, tok, alu: XRayALUImplant, device: torch.device) -> Dict[str, bool]:
    """Check (1) silent prose bit-identical logits, (2) pre-trigger bit-identical logits."""
    prose = (
        "The observatory sits atop a granite ridge overlooking the pine valley. "
        "Every winter astronomers calibrate the mirrors before dawn."
    )
    x = torch.tensor([tok.encode(prose, add_special_tokens=False)], device=device)

    alu.detach()
    logits_plain = model(input_ids=x).logits

    alu.attach()
    alu.prime(x)
    silent_flag = not bool(alu.active_flag.any())
    logits_with = model(input_ids=x).logits
    prose_exact = bool(torch.equal(logits_plain, logits_with))

    # Also verify step-by-step pre-trigger identity on a real question!
    prompt = FEWSHOT_ALU["count"] + "Q: How many times does the letter 'r' appear in 'strawberry'?\nA:"
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)

    alu.enabled = False
    plain_ids, _, plain_lg = alu.generate_greedy(ids, max_new_tokens=28, stop_on_newline=True)

    alu.enabled = True
    imp_ids, written, imp_lg = alu.generate_greedy(ids, max_new_tokens=28, stop_on_newline=True)

    first_write = written.index(True) if True in written else len(written)
    pre_trigger_exact = (
        first_write > 0
        and first_write < len(written)
        and imp_ids[:first_write] == plain_ids[:first_write]
        and all(torch.equal(a, b) for a, b in zip(plain_lg[:first_write], imp_lg[:first_write]))
    )
    decoded_plain = tok.decode(plain_ids).strip()
    decoded_imp = "".join(
        f"[{tok.decode([t])}]" if w else tok.decode([t]) for t, w in zip(imp_ids, written)
    ).strip()
    print(f"[Verify] Silent prose active={not silent_flag}, bit_identical={prose_exact}")
    print(f"[Verify] Pre-trigger bit_identical for {first_write} tokens: {pre_trigger_exact}")
    print(f"[Verify] Demo plain  : {decoded_plain!r}")
    print(f"[Verify] Demo implant: {decoded_imp!r}")
    return {
        "prose_silent": silent_flag,
        "prose_bit_identical": prose_exact,
        "pre_trigger_bit_identical": pre_trigger_exact,
        "pre_trigger_steps": first_write,
        "demo_plain": decoded_plain,
        "demo_implant": decoded_imp,
    }


def extract_answer(text: str, is_alu: bool) -> str:
    line = text.strip().split("\n")[0].strip()
    if is_alu and "]=" in line:
        line = line.split("]=", 1)[1].strip()
    # Strip trailing period/punctuation if any
    line = line.rstrip(".,;:")
    return line.lower().strip()


@torch.no_grad()
def eval_examples(
    model,
    tok,
    alu: XRayALUImplant,
    examples: List[CharExample],
    mode: str,  # "plain" | "alu_icl" | "alu_direct"
    device: torch.device,
) -> Tuple[float, List[Dict]]:
    correct = 0
    rows = []
    for ex in examples:
        q_line = ex.prompt.replace("\nAnswer:", "")
        if mode == "plain":
            prompt = FEWSHOT_PLAIN[ex.task] + f"Q: {q_line}\nA:"
            alu.enabled = False
        elif mode == "alu_icl":
            prompt = FEWSHOT_ALU[ex.task] + f"Q: {q_line}\nA:"
            alu.enabled = True
        elif mode == "alu_direct":
            assert ex.alu_call is not None
            prompt = ex.alu_call
            alu.enabled = True
        else:
            raise ValueError(mode)

        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)
        max_toks = 40 if ex.task in ("spell", "reverse", "rot13") else 28
        gen_ids, written, _ = alu.generate_greedy(ids, max_new_tokens=max_toks, stop_on_newline=True)
        raw = tok.decode(gen_ids)
        pred = extract_answer(raw, is_alu=(mode == "alu_icl"))
        ok = (pred == ex.answer.lower())
        if ok:
            correct += 1
        rows.append({
            "task": ex.task,
            "split": ex.split_tag,
            "word": ex.word,
            "gold": ex.answer,
            "pred": pred,
            "raw": raw.strip(),
            "fired": any(written),
            "n_written": sum(written),
            "ok": ok,
        })
    acc = correct / max(1, len(examples))
    return acc, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n_per_task", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.dtype == "bfloat16" and device.type == "cuda") else torch.float32
    print(f"=== BPE X-Ray ALU Eval | model={args.model} | device={device} | dtype={dtype} ===")

    assert verify_relu_digits(device), "ReLU digit extractor failed!"
    print("[Verify] ReLUDigitExtractor exact on all 1,000 integers (0..999): PASS")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    t_load = time.time() - t0

    t1 = time.time()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    bank = CharWeightBank.from_tokenizer(tok, vocab_size=vocab_size, verbose=True)
    alu = XRayALUImplant(model, tok, bank=bank, device=device).attach()
    t_compile = time.time() - t1
    print(f"[Compile] Model load={t_load:.1f}s | X-Ray ALU auto-compile={t_compile:.2f}s")

    verify_info = verify_bit_identical(model, tok, alu, device)

    _, test_words = get_word_splits(seed=args.seed)
    test_nonce = generate_nonce_words(200, seed=args.seed + 200)
    rng = random.Random(args.seed + 55)

    tasks = ["count", "length", "nth", "spell", "reverse", "rot13"]
    suites: Dict[str, List[CharExample]] = {}
    for task in tasks:
        suites[f"{task}_real"] = [
            make_example(task, test_words[i % len(test_words)], rng, "real")
            for i in range(args.n_per_task)
        ]
        suites[f"{task}_nonce"] = [
            make_example(task, test_nonce[i % len(test_nonce)], rng, "nonce")
            for i in range(args.n_per_task)
        ]
    suites["count_multiword"] = make_multiword_count_examples(test_words, n=args.n_per_task, seed=args.seed + 300)

    summary: Dict[str, Dict[str, float]] = {}
    sample_demos: Dict[str, List[Dict]] = {}

    for suite_name, exs in suites.items():
        acc_plain, rows_plain = eval_examples(model, tok, alu, exs, "plain", device)
        acc_icl, rows_icl = eval_examples(model, tok, alu, exs, "alu_icl", device)
        acc_dir, rows_dir = eval_examples(model, tok, alu, exs, "alu_direct", device)
        fire_rate = sum(r["fired"] for r in rows_icl) / len(rows_icl)
        summary[suite_name] = {
            "plain_2shot": round(acc_plain * 100, 2),
            "alu_2shot_icl": round(acc_icl * 100, 2),
            "alu_direct_exact": round(acc_dir * 100, 2),
            "icl_fire_rate": round(fire_rate * 100, 2),
        }
        sample_demos[suite_name] = rows_icl[:5]
        print(
            f"{suite_name:18s} | Plain 2-shot: {acc_plain*100:5.1f}% | "
            f"+XRay ALU (2-shot): {acc_icl*100:5.1f}% (fired {fire_rate*100:4.0f}%) | "
            f"ALU Direct: {acc_dir*100:5.1f}%"
        )

    # Aggregate means
    real_keys = [f"{t}_real" for t in tasks]
    nonce_keys = [f"{t}_nonce" for t in tasks]
    agg = {
        "real_mean": {
            m: round(sum(summary[k][m] for k in real_keys) / len(real_keys), 2)
            for m in ("plain_2shot", "alu_2shot_icl", "alu_direct_exact")
        },
        "nonce_mean": {
            m: round(sum(summary[k][m] for k in nonce_keys) / len(nonce_keys), 2)
            for m in ("plain_2shot", "alu_2shot_icl", "alu_direct_exact")
        },
    }
    print("\n=== AGGREGATE MEANS (6 tasks) ===")
    print(f"Held-out Real Words : Plain={agg['real_mean']['plain_2shot']:.1f}% -> +ALU(2-shot)={agg['real_mean']['alu_2shot_icl']:.1f}% | Direct={agg['real_mean']['alu_direct_exact']:.1f}%")
    print(f"Unseen Nonce Words  : Plain={agg['nonce_mean']['plain_2shot']:.1f}% -> +ALU(2-shot)={agg['nonce_mean']['alu_2shot_icl']:.1f}% | Direct={agg['nonce_mean']['alu_direct_exact']:.1f}%")

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    out_path = args.out or f"results/alu_eval_{slug}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "model": args.model,
        "n_per_task": args.n_per_task,
        "compile_seconds": round(t_compile, 3),
        "verify": verify_info,
        "summary": summary,
        "aggregate": agg,
        "sample_demos": sample_demos,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
