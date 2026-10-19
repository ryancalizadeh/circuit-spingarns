"""
Fixed-point residual against iteration on the ideal diode on its own branch (best_comparisons.md, rank 1) at 256x256:
fixed matching (gamma = R_min, alpha = 0.8) against adaptive matching (alpha = 2/3), one run each (seed 0), with the
iterations at which adaptive re-matches marked and labelled by how many branches it re-matched. Reads the stored
histories of results/best-1d1r-internal; nothing is re-run.

    python -X utf8 internal_diode_residual_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE

RUN_DIR = Path("results/best-1d1r-internal")
OUT = Path("report_assets/best_comparisons/1d1r_internal_residual_256.png")
SIZE, SEED = 256, 0
STYLE = {  # label, color; the slots adaptive_matching_figures.py uses
    "spingarn": ("fixed γ = R_min (α = 0.8)", SERIES_COLORS[0]),
    "spingarn_adaptive": ("adaptive matching (α = 2/3)", SERIES_COLORS[6]),
}


def main():
    with open(RUN_DIR / "runs.jsonl") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    run = {r["algorithm"]: r for r in runs if r["p"] == SIZE and r["seed"] == SEED and r["algorithm"] in STYLE}

    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor=SURFACE)
    for alg, (label, color) in STYLE.items():
        h = np.asarray(run[alg]["history"]["fixed_point_history"])
        k = np.arange(1, h.size + 1)
        ax.plot(k, h, color=color, lw=2, label=f"{label}: {h.size} iterations, {run[alg]['runtime']:.1f} s")
        ax.annotate(f"{h.size}", (k[-1], h[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8.5)

    # each re-match happens before the network step of iteration at (0-based), so h[at] is the first residual under
    # the new matching; plotted at iteration at + 1 like the curve
    alg = "spingarn_adaptive"
    h = np.asarray(run[alg]["history"]["fixed_point_history"])
    color = STYLE[alg][1]
    for at, changed, _ in run[alg]["history"]["rematch_log"]:
        ax.axvline(at + 1, color=color, alpha=0.12, lw=6, zorder=0)
        ax.plot(at + 1, h[at], ls="none", marker="o", ms=8, mfc=SURFACE, mec=color, mew=1.8, zorder=3)
        ax.annotate(f"{changed:,}", (at + 1, h[at]), xytext=(7, 7), textcoords="offset points", ha="left",
                    va="bottom", color=INK_2, fontsize=7.5)
    ax.plot([], [], ls="none", marker="o", ms=8, mfc=SURFACE, mec=color, mew=1.8,
            label="re-match + refactorization (label: branches re-matched)")

    # reference: linear convergence at rate 1/3 (alpha = 2/3 on the -1 mode), through adaptive's last point, drawn back
    # to where it reaches 1e-1; on top, since adaptive follows it from ~iteration 15
    end = h.size
    start = end - np.log(0.1 / h[-1]) / np.log(3)
    k_ref = np.linspace(start, end, 200)
    ax.plot(k_ref, h[-1] * 3.0 ** (end - k_ref), color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=4,
            label="reference: rate 1/3 per iteration, through adaptive's last point")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor(SURFACE)
    ax.set_title(f"Ideal diode on its own branch, {SIZE}×{SIZE} (seed {SEED}): fixed-point residual per iteration",
                 loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel("iteration", color=INK_2)
    ax.set_ylabel("relative fixed-point residual", color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=2,
               loc="lower center", handlelength=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
