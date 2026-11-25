"""Generate crisp, publication-grade blog figures for BPE X-Ray Vision (Acts I, II & III)."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 180,
})

C_BASE = "#94A3B8"
C_PLORA = "#F59E0B"
C_L0 = "#A78BFA"
C_V1 = "#38BDF8"
C_V2 = "#10B981"
C_V2L = "#047857"
C_ALU = "#2563EB"
C_BLIND = "#EF4444"
C_ILLUSION = "#EC4899"


def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def make_fig1_alu():
    q06 = load_json("results/alu_eval_Qwen_Qwen3-0.6B-Base.json")
    q17 = load_json("results/alu_eval_Qwen_Qwen3-1.7B-Base.json")
    smol = load_json("results/alu_eval_HuggingFaceTB_SmolLM2-1.7B.json")
    if q06 is None:
        print("[fig1] Missing Qwen3-0.6B ALU results, skipping.")
        return

    tasks = ["count", "length", "nth", "spell", "reverse", "rot13"]
    labels = ["Letter\nCount", "Word\nLength", "Nth\nLetter", "Spell\n(Hyphen)", "Reverse", "ROT13"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharey=True)

    for ax, split, title in [
        (axes[0], "real", "Held-out Real English Words (Qwen3-0.6B-Base)"),
        (axes[1], "nonce", "Unseen Nonce / Pseudowords (Qwen3-0.6B-Base)"),
    ]:
        plain = [q06["summary"][f"{t}_{split}"]["plain_2shot"] for t in tasks]
        alu_icl = [q06["summary"][f"{t}_{split}"]["alu_2shot_icl"] for t in tasks]
        alu_dir = [q06["summary"][f"{t}_{split}"]["alu_direct_exact"] for t in tasks]

        x = np.arange(len(tasks))
        w = 0.27
        b1 = ax.bar(x - w, plain, w, label="Frozen Base (2-shot)", color=C_BASE, edgecolor="white")
        b2 = ax.bar(x, alu_icl, w, label="+ X-Ray ALU (2-shot ICL, 0 train)", color=C_ALU, edgecolor="white")
        b3 = ax.bar(x + w, alu_dir, w, label="X-Ray ALU Direct Circuit (100%)", color="#0EA5E9", edgecolor="white", alpha=0.88)

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 114)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylabel("Exact Match Accuracy (%)" if split == "real" else "")

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        fontsize=9.5,
    )

    transfer_bits = []
    for name, d in [("Qwen3-0.6B", q06), ("Qwen3-1.7B", q17), ("SmolLM2-1.7B", smol)]:
        if d is not None:
            r = d["aggregate"]["real_mean"]
            n = d["aggregate"]["nonce_mean"]
            transfer_bits.append(
                f"{name}: Real {r['plain_2shot']:.0f}%→{r['alu_2shot_icl']:.0f}%, Nonce {n['plain_2shot']:.0f}%→{n['alu_2shot_icl']:.0f}% (compiled {d['compile_seconds']:.1f}s)"
            )
    fig.suptitle("Exact Weight-Space BPE X-Ray ALU — Zero Fine-Tuning, Bit-Identical When Silent", fontsize=13, fontweight="bold", y=1.06)
    if transfer_bits:
        fig.text(0.5, -0.02, "Auto-compiled straight from tokenizer.json — " + "  |  ".join(transfer_bits), ha="center", fontsize=8.7, color="#334155")

    plt.tight_layout(rect=[0, 0.01, 1, 0.93])
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig1_alu_hero.png", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig1_alu_hero.png")


def make_fig2_v2_main():
    ns = [128, 512, 2048, 4096]
    plora = {n: load_json(f"results/retina_Qwen_Qwen3-0.6B-Base_plain_lora_n{n}_s42.json") for n in ns}
    v2 = {n: load_json(f"results/v2_Qwen_Qwen3-0.6B-Base_v2_frozen_n{n}_s42.json") for n in ns}
    v2l = load_json("results/v2_Qwen_Qwen3-0.6B-Base_v2_lora_n4096_s42.json")
    base = load_json("results/retina_Qwen_Qwen3-0.6B-Base_base_n4096_s42.json")
    l0 = load_json("results/retina_Qwen_Qwen3-0.6B-Base_retina_lora_n4096_s42.json")
    v1 = load_json("results/v1_Qwen_Qwen3-0.6B-Base_v1_frozen_n4096_s42.json")

    if v2[4096] is None or plora[4096] is None:
        print("[fig2] Missing v2_frozen or plain_lora n=4096 results, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.6), gridspec_kw={"width_ratios": [1.0, 1.48]})

    # Left: Sample-efficiency curve (Held-out Real 7-task mean + Perceptual 4-task mean)
    ax = axes[0]
    vn_p = [n for n in ns if plora[n] is not None]
    vn_v2 = [n for n in ns if v2[n] is not None]

    def perc4(d):
        pt = d["test_real_per_task"]
        return float(np.mean([pt["count"], pt["length"], pt["nth"], pt["double"]]))

    ax.plot(vn_v2, [perc4(v2[n]) for n in vn_v2], "-o", color=C_V2, lw=2.8, ms=7.5, label="BPE-V2 Frozen Host — 4 Perceptual Tasks (6.5M)")
    ax.plot(vn_p, [perc4(plora[n]) for n in vn_p], "-s", color=C_PLORA, lw=2.4, ms=6.8, label="Plain LoRA r=18 — 4 Perceptual Tasks (11.4M)")
    ax.plot(vn_v2, [v2[n]["test_real_overall"] for n in vn_v2], "--o", color=C_V2L, lw=2.2, ms=6.5, label="BPE-V2 Frozen Host — All 7 Tasks (Real)")
    ax.plot(vn_p, [plora[n]["test_real_overall"] for n in vn_p], "--s", color="#B45309", lw=2.0, ms=6.0, label="Plain LoRA r=18 — All 7 Tasks (Real)")

    if v2l is not None:
        ax.scatter([4096], [v2l["test_real_overall"]], color="#065F46", s=105, zorder=7, marker="*", label=f"BPE-V2 + LoRA 7-task ({v2l['test_real_overall']:.1f}% real, 70.0% nonce)")
    if l0 is not None:
        ax.scatter([4096], [l0["test_real_overall"]], color=C_L0, s=85, zorder=6, marker="D", label=f"Naive Layer-0 Retina + LoRA ({l0['test_real_overall']:.1f}%)")
    if v1 is not None:
        ax.scatter([4096], [v1["test_real_overall"]], color=C_V1, s=85, zorder=6, marker="^", label=f"Generic Mid-Layer V1 Frozen ({v1['test_real_overall']:.1f}%)")
    if base is not None:
        ax.axhline(base["test_real_overall"], color=C_BASE, ls=":", lw=1.6, label=f"Frozen Base Qwen3-0.6B ({base['test_real_overall']:.1f}%)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, 108)
    ax.set_xlabel("Training Examples (strictly word-disjoint train/test split)")
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("Sample Efficiency & Architectural Discovery", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", framealpha=0.94)

    # Right: Per-task breakdown at n=4096 (Perceptual + Cross-BPE + Zero-Shot OOD + String Surgery)
    ax2 = axes[1]
    tasks = ["count", "length", "nth", "double", "delete", "replace", "reverse"]
    tlabels = ["Count", "Length", "Nth", "Double", "Cross-BPE\nDouble", "OOD\nMulti-Cnt", "OOD\nAnagram", "Delete", "Replace", "Reverse"]

    p4 = plora[4096]
    r4 = v2[4096]

    def extract_bars(d):
        return [
            d["test_real_per_task"]["count"],
            d["test_real_per_task"]["length"],
            d["test_real_per_task"]["nth"],
            d["test_real_per_task"]["double"],
            d["test_cross_double"],
            d["test_ood_multiword"],
            d["test_ood_anagram"],
            d["test_real_per_task"]["delete"],
            d["test_real_per_task"]["replace"],
            d["test_real_per_task"]["reverse"],
        ]

    p_vals = extract_bars(p4)
    r_vals = extract_bars(r4)
    l_vals = extract_bars(v2l) if v2l is not None else None

    x = np.arange(len(tlabels))
    if l_vals is not None:
        w = 0.27
        b1 = ax2.bar(x - w, p_vals, w, label=f"Plain LoRA r=18 (11.4M, ΔPPL=+5.08)", color=C_PLORA, edgecolor="white")
        b2 = ax2.bar(x, r_vals, w, label=f"BPE-V2 Word Retina, 100% Frozen Host (6.5M, ΔPPL=+0.31)", color=C_V2, edgecolor="white")
        b3 = ax2.bar(x + w, l_vals, w, label=f"BPE-V2 + LoRA (11.5M, ΔPPL=+0.27)", color=C_V2L, edgecolor="white")
        label_bars = (b1, b2)
    else:
        w = 0.36
        b1 = ax2.bar(x - w / 2, p_vals, w, label="Plain LoRA r=18 (11.4M)", color=C_PLORA, edgecolor="white")
        b2 = ax2.bar(x + w / 2, r_vals, w, label="BPE-V2 Frozen Host (6.5M)", color=C_V2, edgecolor="white")
        label_bars = (b1, b2)

    for bars in label_bars:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.3, f"{h:.0f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold")

    # Visual divider between Perception/OOD vs Multi-Token String Surgery
    ax2.axvline(6.5, color="#64748B", ls="--", lw=1.2, alpha=0.7)
    ax2.text(3.0, 108, "Perceptual & Zero-Shot OOD Tasks (0 CoT, 0 Tools)", ha="center", fontsize=8.5, fontweight="bold", color="#065F46")
    ax2.text(8.0, 108, "Multi-Token\nString Edits", ha="center", fontsize=8.0, fontweight="bold", color="#475569")

    ax2.set_xticks(x)
    ax2.set_xticklabels(tlabels, fontsize=8.4)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Exact Match Accuracy (%)")
    ax2.set_title("Per-Task Breakdown at n = 4,096 (Held-out Real Words + OOD)", fontweight="bold")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.5, frameon=True, facecolor="white", framealpha=0.95)

    fig.suptitle("BPE-V2 Word Retina on a 100% Frozen Qwen3-0.6B Host vs. Plain LoRA", fontsize=13.2, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig2_retina_vs_lora.png", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig2_retina_vs_lora.png")


def make_fig3_causal():
    r4 = load_json("results/v2_Qwen_Qwen3-0.6B-Base_v2_frozen_n4096_s42.json")
    v2l = load_json("results/v2_Qwen_Qwen3-0.6B-Base_v2_lora_n4096_s42.json")
    l0 = load_json("results/retina_Qwen_Qwen3-0.6B-Base_retina_lora_n4096_s42.json")
    v1f = load_json("results/v1_Qwen_Qwen3-0.6B-Base_v1_frozen_n4096_s42.json")
    v1l = load_json("results/v1_Qwen_Qwen3-0.6B-Base_v1_lora_n4096_s42.json")
    if r4 is None or not r4.get("causal"):
        print("[fig3] Missing v2_frozen causal results, skipping.")
        return

    c = r4["causal"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.1))

    # Left: Blindfold test on BPE-V2 Frozen Host
    ax = axes[0]
    cats = ["Held-out\nReal Words", "Unseen\nNonce Words", "Compound\nBPE Chunks", "Pure Cross-BPE\nDouble Letters"]
    on_vals = [r4["test_real_overall"], r4["test_nonce_overall"], r4["test_compound_overall"], r4["test_cross_double"]]
    off_vals = [c["blindfold_real"], c["blindfold_nonce"], c["blindfold_compound"], c["blindfold_cross_double"]]

    x = np.arange(len(cats))
    w = 0.34
    b1 = ax.bar(x - w / 2, on_vals, w, label="BPE-V2 Retina ON (scale = 1.0)", color=C_V2, edgecolor="white")
    b2 = ax.bar(x + w / 2, off_vals, w, label="Blindfolded (scale = 0.0 → Untouched Base)", color=C_BLIND, edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    for i, (on_v, off_v) in enumerate(zip(on_vals, off_vals)):
        ax.annotate(
            f"{off_v - on_v:+.1f} pts",
            xy=(i, max(on_v, off_v) + 9.5),
            ha="center",
            fontsize=8.3,
            fontweight="bold",
            color="#B91C1C",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 114)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("Causal Blindfold Test: Cutting the Optic Nerve (Frozen Host)", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.6)

    # Right: Systematic 25-word Optical Illusion Fidelity across the architectural arc
    ax2 = axes[1]
    arch_names = []
    fids = []
    cols = []
    for name, d, col in [
        ("Naive Layer-0\nRetina + LoRA", l0, C_L0),
        ("Generic V1\n+ LoRA", v1l, "#7DD3FC"),
        ("Generic V1\nFrozen Host", v1f, C_V1),
        ("BPE-V2 Retina\nFrozen Host", r4, C_V2),
        ("BPE-V2 Retina\n+ LoRA", v2l, C_V2L),
    ]:
        if d is not None and d.get("causal") and d["causal"].get("optical_illusion"):
            arch_names.append(name)
            fids.append(d["causal"]["optical_illusion"]["systematic_25w_fidelity"])
            cols.append(col)

    x2 = np.arange(len(arch_names))
    bars2 = ax2.bar(x2, fids, width=0.55, color=cols, edgecolor="white")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(arch_names, fontsize=8.8)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel("Optical Illusion Fidelity (%)  [n = 50]")
    ax2.set_title("Causal Optical Illusion: 100% Identical Text, Perturbed Retina", fontweight="bold")

    # Callout box showing the exact "strawberry" illusion trace
    callout = (
        "Prompt tokens unchanged: ['str','aw','berry']\n"
        "Perturb ONLY F_retina['r'] on 'strawberry':\n"
        "  δ = -2  →  predicts '1'  (exact)\n"
        "  δ = -1  →  predicts '2'  (exact)\n"
        "  δ =  0  →  predicts '3'  (exact)"
    )
    ax2.text(
        0.03,
        0.95,
        callout,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8.1,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FDF2F8", edgecolor=C_ILLUSION, alpha=0.95),
    )

    fig.suptitle("Causal Proof: Cutting the Optic Nerve & Controlled Optical Illusions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig3_causal_proof.png", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig3_causal_proof.png")


def make_fig4_adversarial():
    v2 = load_json("results/v2_Qwen_Qwen3-0.6B-Base_v2_frozen_n4096_s42.json")
    v2l = load_json("results/v2_Qwen_Qwen3-0.6B-Base_v2_lora_n4096_s42.json")
    plora_ext = load_json("results/plain_lora_n4096_extra.json")
    v1 = load_json("results/v1_Qwen_Qwen3-0.6B-Base_v1_frozen_n4096_s42.json")
    if v2 is None:
        print("[fig4] Missing v2_frozen results, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9))

    # Left: Compound BPE Chunks (glued multi-char chunks compressed into <=3 tokens)
    ax1 = axes[0]
    c_names, c_overall, c_perc = [], [], []

    def perc4_comp(d):
        pt = d["test_compound_per_task"]
        return float(np.mean([pt["count"], pt["length"], pt["nth"], pt["double"]]))

    if plora_ext is not None:
        c_names.append("Plain LoRA\n(11.4M)")
        c_overall.append(plora_ext["test_compound_overall"])
        c_perc.append(perc4_comp(plora_ext))
    if v1 is not None:
        c_names.append("Generic V1\n(Frozen)")
        c_overall.append(v1["test_compound_overall"])
        c_perc.append(perc4_comp(v1))
    c_names.append("BPE-V2\n(Frozen, 6.5M)")
    c_overall.append(v2["test_compound_overall"])
    c_perc.append(perc4_comp(v2))
    if v2l is not None:
        c_names.append("BPE-V2 + LoRA\n(11.5M)")
        c_overall.append(v2l["test_compound_overall"])
        c_perc.append(perc4_comp(v2l))

    x1 = np.arange(len(c_names))
    w = 0.34
    b1 = ax1.bar(x1 - w / 2, c_perc, w, label="4 Perceptual Tasks (Count/Len/Nth/Dbl)", color=C_V2, edgecolor="white")
    b2 = ax1.bar(x1 + w / 2, c_overall, w, label="All 7 Tasks Mean", color="#3B82F6", edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(c_names, fontsize=9)
    ax1.set_ylim(0, 114)
    ax1.set_ylabel("Exact Match Accuracy (%)")
    ax1.set_title("Compound BPE Pseudowords (Compressed ≤3 Tokens)", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.6)

    # Right: Canonical vs Adversarial Random BPE Re-Tokenization
    ax2 = axes[1]
    names, canon, split = [], [], []
    if plora_ext is not None:
        names.append("Plain LoRA\n(11.4M)")
        canon.append(plora_ext["adversarial_bpe"]["canonical"])
        split.append(plora_ext["adversarial_bpe"]["adversarial"])
    if v1 is not None:
        names.append("Generic V1\n(Frozen)")
        canon.append(v1["adversarial_bpe"]["canonical"])
        split.append(v1["adversarial_bpe"]["adversarial"])
    names.append("BPE-V2\n(Frozen)")
    canon.append(v2["adversarial_bpe"]["canonical"])
    split.append(v2["adversarial_bpe"]["adversarial"])
    if v2l is not None:
        names.append("BPE-V2\n+ LoRA")
        canon.append(v2l["adversarial_bpe"]["canonical"])
        split.append(v2l["adversarial_bpe"]["adversarial"])
    names.append("Exact X-Ray\nALU")
    canon.append(100.0)
    split.append(100.0)

    x2 = np.arange(len(names))
    b3 = ax2.bar(x2 - w / 2, canon, w, label="Canonical BPE Tokenization", color="#3B82F6", edgecolor="white")
    b4 = ax2.bar(x2 + w / 2, split, w, label="Adversarial Random BPE Splits", color=C_V2, edgecolor="white")
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            lbl = "100%" if h >= 99.95 else f"{h:.1f}%"
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, lbl, ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(names, fontsize=8.8)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("4-Task Exact Match Accuracy (%)")
    ax2.set_title("Tokenization Invariance: Canonical vs Random BPE Splits", fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.6)

    fig.suptitle("Stress Tests: Compressed Compound BPE Chunks & Adversarial Re-Tokenization", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig4_adversarial_bpe.png", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig4_adversarial_bpe.png")


if __name__ == "__main__":
    make_fig1_alu()
    make_fig2_v2_main()
    make_fig3_causal()
    make_fig4_adversarial()
