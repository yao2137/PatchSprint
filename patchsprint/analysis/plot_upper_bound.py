"""Figures for the copyable-output analysis (reads results of copy_upper_bound.py)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {  # categorical slots 1-3, validated as a set (CVD worst adjacent dE 9.2)
    "ngram": ("vLLM-style n-gram", "#2a78d6", "o"),
    "pointer": ("Pointer (ours)", "#eb6834", "s"),
    "hybrid": ("Pointer + n-gram fallback (ours)", "#1baf7a", "D"),
}
BOUND = ("Copy oracle (upper bound)", TEXT_2)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_copy_fraction(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.0), facecolor=SURFACE)
    style_axes(ax)
    vals = df["copy_frac_m8"] * 100
    ax.hist(vals, bins=range(0, 105, 5), color="#2a78d6", edgecolor=SURFACE, linewidth=2)
    med = vals.median()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, top)
    ax.axvline(med, color=TEXT, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(med - 1.5, top * 0.97, f"median {med:.0f}%", ha="right", va="top",
            fontsize=9, color=TEXT)
    ax.set_xlabel("Output tokens copied from the input in spans of 8+ tokens (%)", fontsize=9, color=TEXT)
    ax.set_ylabel("Problems", fontsize=9, color=TEXT)
    ax.set_title(f"How much of an edited file is copied? (CanItEdit, n={len(df)})",
                 fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_speedup_vs_cost(summary: dict, ks: list[int], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), facecolor=SURFACE)
    panels = [
        ("aggregate_step_speedup", "Decode-step speedup (x)",
         "(a) Fewer forward passes\n(benefit when verification is memory-bound)"),
        ("verified_tokens_per_output_token", "Tokens verified per output token",
         "(b) Extra tokens pushed through the model\n(cost when verification is compute-bound)"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        style_axes(ax)
        bound = [summary[f"ctx_oracle_k{k}"][metric] for k in ks]
        ax.plot(ks, bound, color=BOUND[1], linewidth=1.5, linestyle=(0, (4, 3)), label=BOUND[0])
        for key, (label, color, marker) in SERIES.items():
            ys = [summary[f"{key}_k{k}"][metric] for k in ks]
            ax.plot(ks, ys, color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        if metric == "verified_tokens_per_output_token":
            ax.text(ks[-1], 1.02, "oracle = plain decoding = 1.0", fontsize=8, color=TEXT_2,
                    va="bottom", ha="right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("Draft budget k (tokens per step)", fontsize=9, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
        ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
        # Direct labels at the right end (values in text ink, identity from the marker).
        last = ks[-1]
        # In panel (b) the oracle line is labelled in place, so skip its end label.
        ends = [] if metric == "verified_tokens_per_output_token" else [(summary[f"ctx_oracle_k{last}"][metric], "oracle")]
        ends += [(summary[f"{key}_k{last}"][metric], key) for key in SERIES]
        # Dodge labels that would collide (minimum gap: 5% of the y range).
        lo, hi = ax.get_ylim()
        gap = 0.05 * (hi - lo)
        placed: list[float] = []
        for y, _ in sorted(ends):
            ly = max(y, placed[-1] + gap) if placed else y
            placed.append(ly)
            ax.annotate(f"{y:.2f}", (last, y), xytext=(last * 1.08, ly), textcoords="data",
                        va="center", fontsize=8, color=TEXT)
        ax.set_xlim(ks[0] / 1.15, ks[-1] * 1.45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=TEXT, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True, help="directory with per_problem.csv and summary.json")
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    args = ap.parse_args()
    df = pd.read_csv(args.results / "per_problem.csv")
    summary = json.loads((args.results / "summary.json").read_text())
    fig_copy_fraction(df, args.results / "fig_copy_fraction.png")
    fig_speedup_vs_cost(summary, args.ks, args.results / "fig_speedup_vs_cost.png")


if __name__ == "__main__":
    main()
