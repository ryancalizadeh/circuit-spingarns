"""
Where the cells of a plain tanh memristor crossbar (no diode) sit on their i-v curve at the Spingarn solution, 8x8 to
256x256: the curve for R_min 1 kohm, R_max 200 kohm against the linear cell i = v / R_min, with each size's median and
90th percentile of |v| over all cells (mean over seeds) marked on the curve. The same sources as the 1D1R sweep
(diode_experiment.CIRCUIT), so the figure is comparable with report_assets/nonsmooth/1d1r_operating_points.png, but with
fixed R_min and R_max. The law is odd, so cells of either polarity are folded onto v > 0.

    python memristor_operating_points_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Config import BranchKind, Config
from algorithms.spingarns import SpingarnSolver
from plotting import INK_2, MUTED, SERIES_COLORS, SURFACE
from report_nonsmooth_figures import style

OUT = Path(__file__).resolve().parent / "report_assets" / "memristor_operating_points.png"
R_MIN, R_MAX = 1e3, 200e3
SIZES = [8, 16, 32, 64, 128, 256]
SEEDS = 3


def operating_point(n, seed):
    cfg = Config(n, n, R_min=R_MIN, R_max=R_MAX, E_range=(-1.0, 1.5), seed=seed, device="memristor")
    out = SpingarnSolver(tol=1e-9, max_iterations=5000).solve(cfg)
    assert out.converged, f"{n}x{n} seed {seed} did not converge"
    x = np.abs(out.v[cfg.slices[BranchKind.DEVICE]])
    print(f"{n}x{n} seed {seed}: {out.iterations} its, |v| median {np.median(x):.4f}, p90 {np.quantile(x, 0.9):.4f}")
    return np.median(x), np.quantile(x, 0.9), cfg


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    v = np.linspace(0, 1.5, 601)
    cell = Config(1, 1, R_min=R_MIN, R_max=R_MAX, seed=0, device="memristor")
    i = cell.device_current(v)
    ax.plot(v, i * 1e3, color=SERIES_COLORS[0], lw=2, label="R_min 1 kΩ, R_max 200 kΩ")
    ax.plot(v, v / R_MIN * 1e3, color=MUTED, lw=1.5, ls="--", label="linear: v = i · R_min")

    top = float(i[-1] * 1e3)
    for k, n in enumerate(SIZES):
        res = [operating_point(n, s) for s in range(SEEDS)]
        med, p90 = np.mean([r[0] for r in res]), np.mean([r[1] for r in res])
        y = top * (0.93 - 0.075 * k)
        ax.plot([med, p90], [y, y], color=INK_2, lw=1.5, solid_capstyle="butt")
        ax.plot(med, y, "o", ms=8, color=INK_2, mec=SURFACE, mew=1.5)
        ax.text(p90 + 0.02, y, f"{n}×{n}", va="center", color=INK_2, fontsize=8.5)
    ax.text(0.03, top * (0.93 + 0.075), "cells in the sweep: median ● to 90th percentile of |v|", color=MUTED,
            fontsize=8.5)
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(-0.05 * top, top * 1.12)
    style(ax, "Single tanh memristor: where the crossbar's cells operate", "cell voltage |v| (V)", "cell current i (mA)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
