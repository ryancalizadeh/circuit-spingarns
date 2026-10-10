"""
Figures for report_nonsmooth.md, written to report_assets/nonsmooth/. The cell characteristics are drawn from the
device laws; every other figure is drawn from, or copied out of, the results of the runs the report describes:

    results/diode-sweep-1          diode_experiment.py (1D1R, alpha sweep)
    results/diode-spice-1          run_experiments.py (1D1R, Spingarn vs SPICE)
    results/selector-feasibility-1 selector_feasibility.py (1S1R, knee sharpness)
    results/selector-scale-Roff*   run_experiments.run_sweep (1S1R at scale)

    python report_nonsmooth_figures.py
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Config import Config
from devices import selector_conductance
from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_assets" / "nonsmooth"
# a typical cell: the geometric middle of the per-cell ranges R_min 1-10 kohm and R_max 50-200 kohm
R_MIN, R_MAX = np.sqrt(1e3 * 1e4), np.sqrt(5e4 * 2e5)
SELECTOR = dict(R_on=1e3, V_th=0.5)


def cell(device, R_min=R_MIN, R_max=R_MAX, **selector):
    """A 1x1 Config holding one cell with these parameters; only its device law is used."""
    return Config(1, 1, R_min=R_min, R_max=R_max, seed=0, device=device, **selector)


def style(ax, title, xlabel, ylabel, zero_lines=True):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    if zero_lines:
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def cell_models():
    """The three cells: memristor and 1D1R (linear current), the selector's conductance, the 1S1R cell (log current)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURFACE)
    v = np.linspace(-1.5, 1.5, 3001)

    ax = axes[0]
    mem = cell("memristor").device_current(v)
    ax.plot(v, v / R_MIN * 1e3, color=MUTED, lw=1.5, ls="--", label="zero-bias slope, i = v / R_min")
    ax.plot(v, mem * 1e3, color=SERIES_COLORS[1], lw=2, label="memristor alone: i = g_m(v)")
    ax.plot(v, cell("1d1r").device_current(v) * 1e3, color=SERIES_COLORS[0], lw=2.5,
            label="1D1R cell: i = g_m(max(v, 0))")
    ax.set_ylim(-0.45, 0.45)
    ax.annotate("diode blocks:\ni = 0 for v ≤ 0", (-0.9, 0), xytext=(-1.4, 0.12), color=INK_2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("kink at v = 0:\nslope 0 → 1/R_min", (0, 0), xytext=(0.25, -0.25), color=INK_2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("tanh saturates:\nslope → 1/R_max", (1.2, mem[np.searchsorted(v, 1.2)] * 1e3), xytext=(0.5, 0.36),
                color=INK_2, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    style(ax, f"(a) Memristor and 1D1R cell (R_min {R_MIN / 1e3:.2f} kΩ, R_max {R_MAX / 1e3:.0f} kΩ)",
          "cell voltage v (V)", "current (mA)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")

    ax = axes[1]
    vs = np.linspace(0, 1, 4001)
    for k, delta in enumerate([0.1, 0.03, 0.01, 0.0]):
        g = selector_conductance(vs, SELECTOR["R_on"], 1e6, SELECTOR["V_th"], delta)
        ax.plot(vs, g * 1e3, color=SERIES_COLORS[k], lw=2, label=f"δ = {delta * 1e3:g} mV" + (" (kink)" if delta == 0 else ""))
    ax.set_yscale("log")
    ax.axvline(SELECTOR["V_th"], color=MUTED, lw=0.8, ls=":")
    ax.text(0.03, 1.5e-3 * 1.3, "off: 1/R_off", color=INK_2, fontsize=8.5)
    ax.text(0.78, 1.0 * 0.55, "on: 1/R_on", color=INK_2, fontsize=8.5)
    ax.text(SELECTOR["V_th"] + 0.01, 3e-3, "V_th", color=INK_2, fontsize=8.5)
    style(ax, "(b) Selector: incremental conductance (R_on 1 kΩ, R_off 1 MΩ)", "selector voltage v_s (V)",
          "conductance (mS)", zero_lines=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center right", title="knee half-width",
              title_fontsize=8)

    ax = axes[2]
    vp = np.linspace(1e-3, 1.5, 1500)
    ax.plot(vp, cell("memristor").device_current(vp) * 1e3, color=SERIES_COLORS[1], lw=2, label="memristor alone")
    for k, R_off in enumerate([1e5, 1e6]):
        i = cell("1s1r", R_off=R_off, delta=0.0, **SELECTOR).device_current(vp)
        ax.plot(vp, i * 1e3, color=SERIES_COLORS[2 + 2 * k], lw=2.5 if k else 2,
                label=f"1S1R cell, R_off = {R_off / 1e3:g} kΩ")
    ax.set_yscale("log")
    ax.annotate("below threshold the selector\ntakes the voltage: i ≈ v / R_off", (0.3, 0.3 / 1e6 * 1e3),
                xytext=(0.12, 2.5e-6), color=INK_2, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("above it the cell conducts\nthrough R_on + memristor", (1.1, 0.12), xytext=(0.82, 1.2e-3),
                color=INK_2, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    style(ax, "(c) 1S1R cell (kink, δ = 0; odd in v)", "cell voltage v (V)", "current (mA, log)", zero_lines=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    save(fig, "cell_models.png")


def operating_points_1d1r():
    """The 1D1R forward branch, normalized by the zero-bias conductance, against where the sweep's cells sit."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    vf = np.linspace(0, 1.5, 601)
    for k, (lo, hi) in enumerate([(1e3, 2e5), (1e3, 5e4), (1e4, 2e5), (1e4, 5e4)]):
        i = cell("1d1r", R_min=lo, R_max=hi).device_current(vf)
        ax.plot(vf, i * lo, color=SERIES_COLORS[k], lw=2,
                label=f"R_min {lo / 1e3:g} kΩ, R_max {hi / 1e3:g} kΩ (R_min/R_max {lo / hi:.3f})")
    ax.plot(vf, vf, color=MUTED, lw=1.5, ls="--", label="linear: i · R_min = v")
    records = [json.loads(line) for line in open(RESULTS / "diode-sweep-1" / "records.jsonl") if line.strip()]
    for k, n in enumerate(sorted({r["p"] for r in records})):
        med = np.mean([r["operating_point"]["v_fwd_median"] for r in records if r["p"] == n])
        p90 = np.mean([r["operating_point"]["v_fwd_p90"] for r in records if r["p"] == n])
        y = 1.38 - 0.075 * k
        ax.plot([med, p90], [y, y], color=INK_2, lw=1.5, solid_capstyle="butt")
        ax.plot(med, y, "o", ms=8, color=INK_2, mec=SURFACE, mew=1.5)
        ax.text(p90 + 0.02, y, f"{n}×{n}", va="center", color=INK_2, fontsize=8.5)
    ax.text(0.03, 1.38 + 0.075, "conducting cells in the sweep: median ● to 90th percentile", color=MUTED, fontsize=8.5)
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(-0.05, 1.5)
    style(ax, "1D1R forward branch, normalized: how far the tanh bends", "cell voltage v (V)",
          "i · R_min (V; equals v for a linear cell)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    save(fig, "1d1r_operating_points.png")


def runtime_panel(ax, run_dirs, solvers, title):
    """Mean runtime vs size per solver, each point labelled with its iteration count; failed sizes drawn at their time
    to failure."""
    runs = [json.loads(line) for d in run_dirs for line in open(d / "runs.jsonl") if line.strip()]
    by = defaultdict(list)
    for r in runs:
        by[(r["algorithm"], r["p"])].append(r)
    sizes = sorted({r["p"] for r in runs})
    data = {}
    for alg, _, _, key in solvers:
        data[alg] = {}
        for n in sizes:
            ok = [r for r in by[(alg, n)] if r["status"] == "converged"]
            if ok:
                its = [r["iterations"] if key == "iterations" else r["history"][key] for r in ok]
                data[alg][n] = (np.mean([r["runtime"] for r in ok]), np.mean(its))
    for alg, label, color, _ in solvers:
        ns = sorted(data[alg])
        ax.plot([n * n for n in ns], [data[alg][n][0] for n in ns], color=color, lw=2, marker="o", ms=8, mec=SURFACE,
                mew=1.5, label=label)
        for n in ns:
            other = [data[o][n][0] for o in data if o != alg and n in data[o]]
            above = not other or data[alg][n][0] >= max(other)
            ax.annotate(f"{data[alg][n][1]:.0f} its", (n * n, data[alg][n][0]), xytext=(0, 10 if above else -16),
                        textcoords="offset points", ha="center", color=INK_2, fontsize=8)
        failed = [r for n in sizes for r in by[(alg, n)] if r["status"] != "converged"]
        if failed:
            n = failed[0]["p"]
            t_fail = np.mean([r["runtime"] for r in failed])
            ax.plot(n * n, t_fail, "x", ms=11, mew=2.2, color=color)
            ax.annotate(f"fails after {t_fail:.0f} s ({len(failed)} of {len(by[(alg, n)])} runs):\nKLU size limit, singular "
                        "matrix\nat the first Newton step", (n * n, t_fail), xytext=(-10, 0), textcoords="offset points",
                        ha="right", va="center", color=INK_2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([n * n for n in sizes], [f"{n}×{n}" for n in sizes])
    ax.minorticks_off()
    style(ax, title, "array size", "total runtime (s), mean of 3 seeds", zero_lines=False)


def comparison_1d1r():
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    runtime_panel(ax, [RESULTS / "diode-spice-1"],
                  [("spingarn", "Spingarn (γ = R_min, α = 0.8)", SERIES_COLORS[0], "iterations"),
                   ("spice", "SPICE (ngspice, KLU)", SERIES_COLORS[1], "spice_iterations")],
                  "1D1R: Spingarn vs SPICE (labels: iterations / Newton iterations)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    save(fig, "1d1r_vs_spice.png")


def scaling_1s1r():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE, sharey=True)
    solvers = [("spingarn_sector", "Spingarn, matched over the sector (α = 1)", SERIES_COLORS[2], "iterations"),
               ("spice", "SPICE (ngspice, KLU)", SERIES_COLORS[1], "spice_iterations")]
    for ax, (R_off, title) in zip(axes, [(100000, "R_off = 100 kΩ"), (1000000, "R_off = 1 MΩ")]):
        runtime_panel(ax, [RESULTS / f"selector-scale-Roff{R_off}"], solvers, f"1S1R, {title} (kink, δ = 0)")
    axes[1].set_ylabel("")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    save(fig, "1s1r_scale.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cell_models()
    operating_points_1d1r()
    comparison_1d1r()
    scaling_1s1r()
    for src, name in [(RESULTS / "diode-sweep-1" / "diode_experiment.png", "1d1r_alpha_sweep.png"),
                      (RESULTS / "selector-feasibility-1" / "selector_feasibility.png", "1s1r_knee.png")]:
        shutil.copyfile(src, OUT / name)
        print(f"copied {src.name} -> {OUT / name}")
