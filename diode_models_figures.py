"""
Figures and summary tables for report_diode_models.md, written to report_assets/diode_models/ and stdout. Every number
comes from the runs the report describes:

    results/diode-models-feasibility    diode_models_experiment.py feasibility (sizes 1-32, all variants, in-process)
    results/diode-models-scale-Is1e-9   diode_models_experiment.py scale, I_s = 1e-9 A, n = 1, all four cases
    results/diode-models-scale-Is1e-6   the same at I_s = 1e-6 A, Shockley cases only

Accuracy: solutions are compared at the word and bit nodes (Config.line_potentials) with Newton on the lumped cell (the
safeguarded variant, which converged in every run); residuals are each solver's own recorded KCL test, absolute, as
relative ones are meaningless on 1D1R's dead lines.

    python -X utf8 diode_models_figures.py
"""
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

from Config import BranchKind, Config
from devices import diode_series_solve, memristor_current
from diode_models_experiment import operating_point, output_change
from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_assets" / "diode_models"
FEASIBILITY = RESULTS / "diode-models-feasibility"
SCALE = {1e-9: RESULTS / "diode-models-scale-Is1e-9", 1e-6: RESULTS / "diode-models-scale-Is1e-6"}

BLUE, ORANGE, TEAL, VIOLET, RED = SERIES_COLORS[0], SERIES_COLORS[1], SERIES_COLORS[2], SERIES_COLORS[6], SERIES_COLORS[7]
# one-hue ordinal ramp (dataviz palette's blue 250/400/550/700), validated --ordinal 2026-10-14
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STYLE = {  # label, colour, line style; blue/orange/teal/violet pass the validator on all pairs (2026-10-14)
    "newton": ("Newton, plain", TEAL, "--"),
    "newton_safeguarded": ("Newton, step halving", TEAL, "-"),
    "spice": ("SPICE, D element", ORANGE, "-"),
    "spice_exp": ("SPICE, behavioural exp", ORANGE, "--"),
    "spice_ron0.001": ("SPICE, uramp / 1 mΩ", ORANGE, "-"),
    "spingarn_fixed": ("Spingarn, fixed γ = R_min (α 0.8)", BLUE, "-"),
    "spingarn_adaptive": ("Spingarn, adaptive (α 2/3)", VIOLET, "-"),
    "spingarn_adaptive_split": ("Spingarn, adaptive split", VIOLET, "--"),
}
CASE_TITLES = {"lumped": "lumped Shockley 1D1R", "internal": "Shockley diode on its own branch",
               "ideal": "ideal diode on its own branch", "ideal_lumped": "ideal lumped 1D1R"}


def load_runs(run_dir):
    path = Path(run_dir) / "runs.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_feasibility():
    with open(FEASIBILITY / "records.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def machine(run_dir):
    return json.loads((Path(run_dir) / "experiment.json").read_text())["machine"]


def result(run_dir, case, alg, p, seed):
    path = Path(run_dir) / "pickles" / f"{case}_{alg}_{p}x{p}_seed{seed}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def style(ax, title, ylabel, logy=True, xlabel="array size"):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    sides = sorted({int(round(np.sqrt(x))) for line in ax.get_lines() for x in line.get_xdata()
                    if np.isfinite(x) and x >= 1 and abs(np.sqrt(x) - round(np.sqrt(x))) < 1e-9})
    if sides:
        ax.set_xticks([k * k for k in sides], [f"{k}×{k}" for k in sides])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def grouped(runs, case):
    """{algorithm: {cells: [runs]}} for one case."""
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["case"] == case:
            out[r["algorithm"]][r["p"] * r["q"]].append(r)
    return out


def iterations_of(r):
    return r["iterations"]


def plot_case(ax, runs, case, value, algorithms):
    """Median over converged seeds per size, each seed a small dot; SPICE runs that needed a fallback are ringed, and
    sizes where some seed failed get an x in a strip above the data, one row per algorithm."""
    groups = grouped(runs, case)
    failed_rows = 0
    for alg in algorithms:
        if alg not in groups:
            continue
        label, color, ls = STYLE[alg]
        xs, ys, fails = [], [], []
        for x in sorted(groups[alg]):
            rs = groups[alg][x]
            ok = [r for r in rs if r["status"] == "converged"]
            if len(ok) < len(rs):
                fails.append(x)
            for r in ok:
                fallback = r["history"].get("method") not in (None, "newton")
                ax.plot(x, value(r), "o", ms=11 if fallback else 3.5, mfc="none" if fallback else color,
                        mec=INK_2 if fallback else color, mew=1.1, alpha=0.9 if fallback else 0.55, zorder=4)
            if ok:
                xs.append(x)
                ys.append(np.median([value(r) for r in ok]))
        if xs:
            ax.plot(xs, ys, color=color, ls=ls, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label, zorder=3)
        if fails:
            ax.plot(fails, [1.03 + 0.045 * failed_rows] * len(fails), "x", ms=8, mew=2, color=color, clip_on=False,
                    transform=ax.get_xaxis_transform())
            failed_rows += 1


def figure_legend(fig, algorithms, top=0.955):
    handles = [Line2D([], [], color=STYLE[a][1], ls=STYLE[a][2], lw=2, marker="o", ms=7, mec=SURFACE, label=STYLE[a][0])
               for a in algorithms]
    handles.append(Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK_2, label="SPICE needed a fallback"))
    handles.append(Line2D([], [], ls="none", marker="x", ms=8, mew=2, color=INK_2, label="a seed did not converge"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top), ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_2, handlelength=3.2)


def legend_note(fig):
    fig.text(0.01, 0.005, "Lines: median of the converged seeds; small dots: each seed. × above a panel: at least one "
             "seed did not converge at that size.", color=MUTED, fontsize=8)


# ---------------------------------------------------------------------------------------------------------------- figures

def cell_iv():
    """The cell laws: ideal 1D1R against lumped Shockley 1D1R for three saturation currents, a typical cell."""
    R_min, R_max, c = 10 ** 3.5, 1e5, 0.02585
    v = np.linspace(-0.6, 1.2, 1201)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    curves = [("ideal diode", memristor_current(np.maximum(v, 0), R_min, R_max), INK, "--")]
    for I_s, color in zip((1e-14, 1e-9, 1e-6), (RAMP[0], RAMP[1], RAMP[3])):
        _, i, _, _, _ = diode_series_solve(v, 0.0, R_min, R_max, I_s, c)
        curves.append((f"Shockley, I_s = {I_s:g} A", i, color, "-"))
    for ax, log in zip(axes, (False, True)):
        for label, i, color, ls in curves:
            y = np.abs(i) if log else i * 1e6
            ax.plot(v, np.where(y > 0, y, np.nan) if log else y, color=color, ls=ls, lw=2, label=label)
        ax.set_facecolor(SURFACE)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-16, 1e-3)
        ax.set_title("|i| on a log scale (reverse current −I_s)" if log else "cell current, linear",
                     loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("cell voltage v (V)", color=INK_2)
        ax.set_ylabel("|i| (A)" if log else "i (µA)", color=INK_2)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    fig.suptitle(f"1D1R cell laws (R_min = {R_min / 1e3:.2f} kΩ, R_max = {R_max / 1e3:g} kΩ, n = 1). The memristor "
                 "caps the forward slope at 1 / R_min;\nthe Shockley diode shifts the turn-on and leaks −I_s in reverse",
                 x=0.01, ha="left", color=INK, fontsize=10.5)
    save(fig, "cell_iv.png")


def operating_points(feas):
    """Regime fractions per size for each diode, from Newton's lumped solution: feasibility sizes, plus the scaling
    sweep's sizes above them for its diodes."""
    diodes = []
    for r in feas:
        if (r["I_s"], r["n"]) not in diodes:
            diodes.append((r["I_s"], r["n"]))
    rows = defaultdict(list)  # (I_s, n, p) -> [operating points]
    changes = defaultdict(list)
    for r in feas:
        rows[r["I_s"], r["n"], r["p"]].append(r["operating_point"])
        changes[r["I_s"], r["n"], r["p"]].append(r["output_change_vs_ideal"])
    for I_s, run_dir in SCALE.items():
        for r in load_runs(run_dir):
            if r["case"] != "lumped" or r["algorithm"] != "newton_safeguarded" or r["p"] <= 32:
                continue
            res = result(run_dir, "lumped", "newton_safeguarded", r["p"], r["seed"])
            ideal = result(SCALE[1e-9], "ideal_lumped", "newton_safeguarded", r["p"], r["seed"])
            if res is None or not res.converged:
                continue
            rows[I_s, 1.0, r["p"]].append(operating_point(res.config, res.v))
            if ideal is not None and ideal.converged:
                changes[I_s, 1.0, r["p"]].append(output_change(ideal.config, ideal.i, res.config, res.i))
    fig, axes = plt.subplots(1, len(diodes), figsize=(4.2 * len(diodes), 4.6), facecolor=SURFACE, sharey=True)
    keys = ("reverse", "subthreshold", "knee", "on")
    labels = ("reverse-biased", "subthreshold (r_d > 10 R_min)", "knee", "on (r_d < R_min / 10)")
    table = []
    for ax, (I_s, n) in zip(np.atleast_1d(axes), diodes):
        sizes = sorted(p for (a, b, p) in rows if (a, b) == (I_s, n))
        x = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for key, label, color in zip(keys, labels, RAMP):
            h = np.array([np.mean([op[key] for op in rows[I_s, n, p]]) for p in sizes])
            ax.bar(x, h, bottom=bottom, color=color, width=0.78, edgecolor=SURFACE, linewidth=2, label=label)
            bottom += h
        ax.set_xticks(x, [f"{p}" for p in sizes])
        ax.set_facecolor(SURFACE)
        ax.set_title(f"I_s = {I_s:g} A, n = {n:g}", loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("array side p (p × p)", color=INK_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
        for p in sizes:
            ops, ch = rows[I_s, n, p], changes[I_s, n, p]
            finite = [c for c in ch if np.isfinite(c)]
            table.append((I_s, n, p, len(ops), *(np.mean([op[k] for op in ops]) for k in keys),
                          min(finite) if finite else np.nan, max(finite) if finite else np.nan,
                          np.median([op["v_diode_fwd_median"] for op in ops if op["v_diode_fwd_median"] is not None])))
    np.atleast_1d(axes)[0].set_ylabel("fraction of diodes (mean of 3 seeds)", color=INK_2)
    handles, labels_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Where the diodes operate at the solution (Newton, lumped cell): knee = forward-biased with r_d within "
                 "a factor 10 of the cell's R_min", x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "operating_points.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'operating_points.png'}")
    print("\nOperating point: I_s, n, size, seeds, reverse, subthreshold, knee, on, output change vs ideal (min-max), "
          "median forward diode voltage")
    for row in table:
        print("  " + " | ".join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)
                                for x in row[:4]) + " | " + " | ".join(f"{x:.2f}" for x in row[4:10]) +
              f" | {row[10]:.3f} V")


def shockley_scale(value, ylabel, name, suffix):
    algorithms = ["newton", "newton_safeguarded", "spice", "spice_exp", "spingarn_fixed", "spingarn_adaptive"]
    sweeps = [(I_s, load_runs(d)) for I_s, d in SCALE.items()]
    sweeps = [(I_s, runs) for I_s, runs in sweeps if runs]
    fig, axes = plt.subplots(len(sweeps), 2, figsize=(14, 4.9 * len(sweeps) + 0.8), facecolor=SURFACE, squeeze=False,
                             sharey="row")
    for row, (I_s, runs) in zip(axes, sweeps):
        for ax, case in zip(row, ("lumped", "internal")):
            plot_case(ax, runs, case, value, algorithms)
            style(ax, f"I_s = {I_s:g} A: {CASE_TITLES[case]}{suffix}", ylabel)
            ax.set_title(ax.get_title(loc="left"), loc="left", color=INK, fontsize=10.5, pad=22)
    figure_legend(fig, algorithms)
    legend_note(fig)
    fig.suptitle(f"Shockley 1D1R, lumped cell (left) against the diode on its own branch (right); measured on "
                 f"{machine(SCALE[1e-9])}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.9))
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def ideal_scale():
    runs = load_runs(SCALE[1e-9])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), facecolor=SURFACE)
    every = ["newton", "newton_safeguarded", "spice_ron0.001", "spingarn_fixed", "spingarn_adaptive",
             "spingarn_adaptive_split"]
    for row, case, algorithms in [(axes[0], "ideal", ["spice_ron0.001", "spingarn_fixed", "spingarn_adaptive"]),
                                  (axes[1], "ideal_lumped", ["newton", "newton_safeguarded", "spice",
                                                             "spingarn_adaptive", "spingarn_adaptive_split"])]:
        for ax, value, what, ylabel in ((row[0], iterations_of, "iterations (SPICE: Newton iterations)", "iterations"),
                                        (row[1], lambda r: r["runtime"], "total runtime", "seconds")):
            plot_case(ax, runs, case, value, algorithms)
            style(ax, f"{CASE_TITLES[case]}: {what}", ylabel)
            ax.set_title(ax.get_title(loc="left"), loc="left", color=INK, fontsize=10.5, pad=22)
    label = STYLE["spice_ron0.001"]
    STYLE["spice_ron0.001"] = ("SPICE, uramp (top: diode R_on = 1 mΩ; bottom: in the memristor source)", ORANGE, "-")
    figure_legend(fig, every)
    STYLE["spice_ron0.001"] = label
    legend_note(fig)
    fig.suptitle(f"Ideal-diode 1D1R: diode on its own branch (top; exact model for Spingarn, R_on = 1 mΩ for SPICE) and "
                 f"lumped (bottom); measured on {machine(SCALE[1e-9])}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.91))
    fig.savefig(OUT / "ideal_scale.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'ideal_scale.png'}")


METHOD_ORDER = ["newton", "gmin stepping", "source stepping", "transient op", "not converged"]
METHOD_COLORS = RAMP + [RED]


def spice_methods(feas):
    """How SPICE reached its operating point, per size: feasibility runs below 64, the scaling sweeps from 8 up (the
    same circuits; where both exist, the sweep's)."""
    panels = [("internal", "spice", 1e-9), ("internal", "spice", 1e-6), ("internal", "spice_exp", 1e-9),
              ("internal", "spice_exp", 1e-6), ("ideal", "spice_ron0.001", None), ("ideal_lumped", "spice", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.2), facecolor=SURFACE, sharey=True)
    for ax, (case, alg, I_s) in zip(axes, panels):
        counts = defaultdict(Counter)
        for r in feas:
            if I_s is not None and (r["I_s"], r["n"]) != (I_s, 1.0):
                continue
            e = r["solves"].get(f"{case}/{alg}")
            if e is None or (I_s is None and (r["I_s"], r["n"]) != (1e-14, 1.0)):
                continue
            counts[r["p"]][e.get("method") if e["status"] == "converged" else "not converged"] += 1
        run_dir = SCALE[I_s if I_s is not None else 1e-9]
        scale_counts = defaultdict(Counter)
        for r in load_runs(run_dir):
            if r["case"] == case and r["algorithm"] == alg:
                m = r["history"].get("method") if r["status"] == "converged" else "not converged"
                scale_counts[r["p"]][m or "not converged"] += 1
        counts.update(scale_counts)
        sizes = sorted(counts)
        x = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for method, color in zip(METHOD_ORDER, METHOD_COLORS):
            h = np.array([counts[p][method] for p in sizes], dtype=float)
            ax.bar(x, h, bottom=bottom, color=color, width=0.78, edgecolor=SURFACE, linewidth=2, label=method,
                   hatch="//" if method == "not converged" else None)
            bottom += h
        ax.set_xticks(x, [str(p) for p in sizes])
        ax.set_facecolor(SURFACE)
        title = {"spice": "uramp, lumped" if case == "ideal_lumped" else "D element", "spice_exp": "behavioural exp",
                 "spice_ron0.001": "uramp / 1 mΩ"}[alg]
        ax.set_title(f"{title}" + (f", I_s = {I_s:g}" if I_s else f"\n{CASE_TITLES[case]}" if case == "ideal_lumped"
                                   else "\nideal, own branch"), loc="left", color=INK, fontsize=9.5)
        ax.set_xlabel("array side p", color=INK_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
    axes[0].set_ylabel("runs (of 3 seeds)", color=INK_2)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=5, frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("How ngspice reached the operating point (ITL1 = 100, RELTOL 1e-3, KLU); 'not converged' includes "
                 "runs whose answer failed the KCL check", x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "spice_methods.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'spice_methods.png'}")


def residual_histories(size=128, seed=0, I_s=1e-9, I_s_newton=1e-6):
    """Spingarn's fixed-point residual (I_s) and Newton's step sizes (I_s_newton, where plain Newton on the internal
    node converged at this size rather than breaking down at its first factorization)."""
    runs = {I: load_runs(SCALE[I]) for I in (I_s, I_s_newton)}
    pick = lambda case, alg, I=I_s: next((r for r in runs[I] if r["case"] == case and r["algorithm"] == alg
                                          and r["p"] == size and r["seed"] == seed), None)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE)
    ax = axes[0]
    for case, ls in (("lumped", "-"), ("internal", "--")):
        for alg, color in (("spingarn_fixed", BLUE), ("spingarn_adaptive", VIOLET)):
            r = pick(case, alg)
            if r is None:
                continue
            h = np.asarray(r["history"]["fixed_point_history"])
            ax.plot(np.arange(1, h.size + 1), h, color=color, ls=ls, lw=2,
                    label=f"{STYLE[alg][0].split(' (')[0]}, {case} ({h.size} its)")
            for k, *_ in r["history"].get("rematch_log", []):
                ax.plot(k + 1, h[k], "o", ms=5, color=color, mec=SURFACE)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(f"Spingarn: fixed-point residual ({size}×{size}, seed {seed}, I_s = {I_s:g} A; dots = re-matches)",
                 loc="left", color=INK, fontsize=10)
    ax.set_xlabel("iteration", color=INK_2)
    ax.set_ylabel("relative fixed-point residual", color=INK_2)
    ax = axes[1]
    for case, alg, ls in (("lumped", "newton", "-"), ("internal", "newton", "--"), ("internal", "newton_safeguarded", ":")):
        r = pick(case, alg, I_s_newton)
        if r is None:
            continue
        h = np.asarray(r["history"]["step_max_history"])
        status = "" if r["status"] == "converged" else f", {r['history'].get('failure') or 'not converged'}"
        ax.plot(np.arange(1, h.size + 1), h, color=TEAL, ls=ls, lw=2, marker="o", ms=4,
                label=f"{STYLE[alg][0]}, {case} ({h.size} steps{status})")
    ax.axhline(0.02585, color=MUTED, lw=1, ls=":")
    ax.text(1.02, 0.02585, "n V_T", color=MUTED, fontsize=8.5, va="bottom")
    ax.set_yscale("log")
    ax.set_title(f"Newton: largest node-potential update per step ({size}×{size}, seed {seed}, I_s = {I_s_newton:g} A)",
                 loc="left", color=INK, fontsize=10)
    ax.set_xlabel("Newton step", color=INK_2)
    ax.set_ylabel("max |Δu| (V)", color=INK_2)
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(True, which="major", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelcolor=INK_2)
        a.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=3.5)
    save(fig, "residual_histories.png")


def ideal_feasibility(feas):
    """Case 3 at small sizes: SPICE's error against the exact solution for each on-resistance, and Spingarn's iterations
    for each relaxation and diode port resistance."""
    recs = [r for r in feas if "ideal/spingarn_adaptive" in r["solves"]]
    sizes = sorted({r["p"] for r in recs})
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=(1, 1.05)))
    ax = axes[0]
    for ron, ls in ((1.0, "-"), (1e-3, "--"), (1e-6, ":")):
        name = f"ideal/spice_ron{ron:g}"
        ys = [max(r["solves"][name].get("max_du", np.nan) for r in recs if r["p"] == p) for p in sizes]
        ax.plot([p * p for p in sizes], ys, color=ORANGE, ls=ls, lw=2, marker="o", ms=7, mec=SURFACE,
                label=f"SPICE, R_on = {ron:g} Ω")
    style(ax, "SPICE on the ideal diode: worst error at word and bit nodes", "max |Δu| over 3 seeds (V)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=3.5)
    ax = axes[1]
    for alpha, color in ((0.5, RAMP[1]), (0.67, RAMP[2]), (0.8, RAMP[3])):
        for port, ls in (("rmin", "-"), ("100", "--"), ("1e5", ":")):
            name = f"ideal/spingarn_a{alpha:.2f}_g{port}"
            ys = []
            for p in sizes:
                its = [r["solves"][name]["iterations"] for r in recs if r["p"] == p
                       and r["solves"][name]["status"] == "converged"]
                ys.append(np.mean(its) if its else np.nan)
            port_label = {"rmin": "R_min", "100": "100 Ω", "1e5": "100 kΩ"}[port]
            ax.plot([p * p for p in sizes], ys, color=color, ls=ls, lw=1.8, marker="o", ms=5, mec=SURFACE,
                    label=f"α = {alpha:g}, diode port {port_label}")
    ys = [np.mean([r["solves"]["ideal/spingarn_adaptive"]["iterations"] for r in recs if r["p"] == p]) for p in sizes]
    ax.plot([p * p for p in sizes], ys, color=VIOLET, lw=2.5, marker="o", ms=7, mec=SURFACE, label="adaptive (α 2/3)")
    style(ax, "Spingarn on the exact ideal diode: iterations", "iterations (mean of converged seeds)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, handlelength=3.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Ideal diode on its own branch, sizes 1×1 to 32×32, 3 seeds (feasibility runs, in-process)",
                 x=0.01, ha="left", color=INK, fontsize=10.5)
    save(fig, "ideal_feasibility.png")


# ----------------------------------------------------------------------------------------------------------------- tables

def table_scale(I_s, cases):
    runs = load_runs(SCALE[I_s])
    if not runs:
        return
    print(f"\n=== Scaling sweep, I_s = {I_s:g} A ({machine(SCALE[I_s])}): iterations range (runtime mean) [SPICE methods]")
    for case in cases:
        groups = grouped(runs, case)
        print(f"  {CASE_TITLES[case]}")
        sizes = sorted({x for g in groups.values() for x in g})
        for alg, by_size in groups.items():
            cells = []
            for x in sizes:
                rs = by_size.get(x, [])
                ok = [r for r in rs if r["status"] == "converged"]
                its = [r["iterations"] for r in ok]
                txt = (f"{min(its)}" if its and min(its) == max(its) else f"{min(its)}-{max(its)}" if its else "-")
                txt += f" ({np.mean([r['runtime'] for r in ok]):.3g} s)" if ok else ""
                if len(ok) < len(rs):
                    txt += f" [{len(rs) - len(ok)} failed]"
                methods = Counter(r["history"].get("method") for r in ok if r["history"].get("method") not in (None,
                                                                                                         "newton"))
                if methods:
                    txt += " [" + ",".join(f"{m[:4]}{k}" for m, k in methods.items()) + "]"
                cells.append(txt)
            print(f"    {alg:<24} " + " | ".join(cells))


def table_accuracy(I_s, cases):
    """Worst max |du| at word and bit nodes against the lumped (safeguarded) Newton answer, and worst recorded absolute
    KCL residual, per solver, over sizes >= 32."""
    run_dir = SCALE[I_s]
    runs = load_runs(run_dir)
    if not runs:
        return
    print(f"\n=== Accuracy, I_s = {I_s:g} A, sizes >= 32: worst max|du| vs lumped Newton (V), worst KCL abs (A)")
    for case in cases:
        ref_case = "ideal_lumped" if case.startswith("ideal") else "lumped"
        ref_dir = SCALE[1e-9] if ref_case == "ideal_lumped" else run_dir
        for alg in sorted({r["algorithm"] for r in runs if r["case"] == case}):
            du, kcl, exact = [], [], []
            for r in runs:
                if r["case"] != case or r["algorithm"] != alg or r["p"] < 32 or r["status"] != "converged":
                    continue
                res = result(run_dir, case, alg, r["p"], r["seed"])
                ref = result(ref_dir, ref_case, "newton_safeguarded", r["p"], r["seed"])
                if res is not None and ref is not None and ref.converged:
                    du.append(np.abs(res.config.line_potentials(res.v) - ref.config.line_potentials(ref.v)).max())
                kcl.append(r["history"].get("kcl_residual", np.nan))
                if "kcl_exact_residual" in r["history"]:
                    exact.append(r["history"]["kcl_exact_residual"])
            if du:
                extra = f", exact-law KCL abs {max(exact):.1e}" if exact else ""
                print(f"  {case:<13} {alg:<24} du {max(du):.1e}  KCL abs {max(kcl):.1e}{extra}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feas = load_feasibility()
    cell_iv()
    operating_points(feas)
    if load_runs(SCALE[1e-9]):
        shockley_scale(iterations_of, "iterations (SPICE: Newton iterations)", "shockley_iterations.png",
                       ": iterations")
        shockley_scale(lambda r: r["runtime"], "total runtime (s)", "shockley_runtime.png", ": runtime")
        ideal_scale()
        residual_histories()
        table_scale(1e-9, ("lumped", "internal", "ideal", "ideal_lumped"))
        table_accuracy(1e-9, ("lumped", "internal", "ideal", "ideal_lumped"))
    if load_runs(SCALE[1e-6]):
        table_scale(1e-6, ("lumped", "internal"))
        table_accuracy(1e-6, ("lumped", "internal"))
    spice_methods(feas)
    ideal_feasibility(feas)


if __name__ == "__main__":
    main()
