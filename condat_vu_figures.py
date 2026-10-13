"""
Figures for report_condat_vu.md, written to report_assets/condat_vu/. The schematic and ||M|| come from the solver's own
code; every other figure is drawn from the results of the runs the report describes:

    results/cv-sweep            run_experiments.py, default crossbar: spingarn, spingarn_sector, condat_vu, spice
    results/cv-best-<n>         run_experiments.py, condat_vu at size n's best rho
    results/cv-sweep-highR      run_experiments.py, high-resistance crossbar (R_min 100 kohm, R_max 10 Mohm, E 1-3 V)
    results/cv-tune, cv-tune-256  condat_vu_experiment.py tune (rho sweeps)

    python condat_vu_figures.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from Config import Config
from algorithms.condat_vu import Interconnect, lines_norm, spanning_tree
from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_assets" / "condat_vu"
SIZES = [8, 16, 32, 64, 128, 256]
STEP_PRODUCT = 0.99
RHOS_TICKS = [150, 200, 300, 400, 500, 700, 1000, 1500, 3000]
# Colors follow the solver everywhere in the repo (plotting.py); condat_vu is pinned to the violet, which passes the
# dataviz validator's all-pairs check beside the other three (worst CVD Delta E 9.2, normal vision 16.3).
BLUE, ORANGE, AQUA, VIOLET = SERIES_COLORS[0], SERIES_COLORS[1], SERIES_COLORS[2], SERIES_COLORS[6]
# Array sizes 8..256 as an ordinal ramp in condat_vu's hue (OKLCH L 0.72 -> 0.33 at constant hue); passes the validator's
# --ordinal checks: monotone, adjacent Delta L >= 0.06, light end 2.48:1 on the surface, hue spread 1 degree.
SIZE_RAMP = ["#9c9be7", "#8482d2", "#6d69bd", "#5851a7", "#433992", "#311e7d"]
SOLVERS = [  # key, label, color, line style
    ("spingarn", "Spingarn, γ = R_min", BLUE, "-"),
    ("spingarn_sector", "Spingarn, γ = √(R_min R_max)", AQUA, "-"),
    ("condat_vu", "Condat–Vũ, ρ = √(R_min R_load)", VIOLET, "-"),
    ("condat_vu_best", "Condat–Vũ, best ρ per size", VIOLET, "--"),
    ("spice", "SPICE (ngspice, KLU)", ORANGE, "-"),
]


def style(ax, title, xlabel, ylabel, logx=False, logy=False):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)


def size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{n}×{n}" for n in sizes])
    ax.minorticks_off()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def runner_runs(circuit="default"):
    """Runner records, keyed (solver key, n); the best-rho Condat-Vu runs become condat_vu_best."""
    out = defaultdict(list)
    if circuit == "default":
        sources = [(RESULTS / "cv-sweep", None)] + [(d, "condat_vu_best") for d in sorted(RESULTS.glob("cv-best-*"))]
    else:
        sources = [(RESULTS / "cv-sweep-highR", None)]
    for d, rename in sources:
        for r in records(d / "runs.jsonl"):
            out[(rename or r["algorithm"], r["p"])].append(r)
    return out


def iterations_of(r):
    return r["history"]["spice_iterations"] if r["algorithm"] == "spice" else r["iterations"]


def mean_over(runs, value):
    ok = [r for r in runs if r["status"] == "converged"]
    return float(np.mean([value(r) for r in ok])) if ok else None


def norm_lines(n):
    return lines_norm(n, n)


def loops_figure():
    """(a) the tree, the links and one cell's fundamental loop on a small crossbar; (b) ||M|| against the side."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    p, q, (R, C) = 4, 5, (1, 2)  # the highlighted cell
    off = 0.32  # bit node offset from its word node
    w = lambda r, c: (c, -r)
    b = lambda r, c: (c + off, -r - off)
    x_src, y_gnd = -1.0, -p - 0.2
    seg = lambda P, Q, **kw: ax.plot([P[0], Q[0]], [P[1], Q[1]], solid_capstyle="round", **kw)
    tree = dict(color=INK_2, lw=1.4, zorder=2)
    loop = dict(color=BLUE, lw=4.2, alpha=0.9, zorder=3)
    # ground rail along the left and the bottom
    seg((x_src, 0.35), (x_src, y_gnd), color=MUTED, lw=1.2, zorder=1)
    seg((x_src, y_gnd), (q - 1 + off + 0.35, y_gnd), color=MUTED, lw=1.2, zorder=1)
    for r in range(p):
        seg((x_src, -r), w(r, 0), **tree)  # source r: E_r and R_source
        ax.plot(x_src + 0.45, -r, marker="o", ms=11, mfc=SURFACE, mec=INK_2, mew=1.4, zorder=4)
        for c in range(q - 1):
            seg(w(r, c), w(r, c + 1), **tree)
    for c in range(q):
        for r in range(p - 1):
            seg(b(r, c), b(r + 1, c), **tree)
        seg(b(p - 1, c), (b(p - 1, c)[0], y_gnd), **tree)  # load c
        ax.plot(b(p - 1, c)[0], (b(p - 1, c)[1] + y_gnd) / 2, marker="s", ms=8, mfc=SURFACE, mec=INK_2, mew=1.4,
                zorder=4)
    for r in range(p):
        for c in range(q):
            shares = r == R or c == C
            seg(w(r, c), b(r, c), color=MUTED if not shares else INK, lw=1.0, ls=(0, (2, 2)), zorder=2)
            ax.plot(*w(r, c), "o", ms=3.5, color=INK_2, zorder=5)
            ax.plot(*b(r, c), "o", ms=3.5, color=INK_2, zorder=5)
    # the fundamental loop of cell (R, C): the cell, its bit line down to the load, ground, the source, the word line
    path = [w(R, C), b(R, C)] + [b(r, C) for r in range(R + 1, p)] + [(b(p - 1, C)[0], y_gnd), (x_src, y_gnd),
                                                                       (x_src, -R)] + [w(R, c) for c in range(C + 1)]
    ax.plot(*zip(*path), **loop)
    right = q + 0.25  # annotations live to the right of the array
    ax.annotate(f"cell ({R}, {C})'s fundamental loop (blue):\nthe cell is the only link on it", (C + off / 2, -R - off / 2),
                xytext=(right, -R + 0.25), va="center", color=INK_2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(right, -R - 1.15, f"every cell on row {R} shares source {R}\nand the word wires up to the nearer cell;\n"
            f"every cell on column {C} shares load {C}\nand the bit wires below the lower cell",
            ha="left", va="top", color=INK_2, fontsize=8.5)
    ax.text(x_src - 0.12, -R, f"source {R}", ha="right", va="center", color=INK_2, fontsize=8.5)
    ax.text(b(p - 1, C)[0] + 0.12, y_gnd + 0.32, f"load {C}", ha="left", va="center", color=INK_2, fontsize=8.5)
    ax.text(x_src, y_gnd - 0.35, "ground", ha="center", va="top", color=MUTED, fontsize=8.5)
    ax.set_xlim(x_src - 1.0, q + 3.4)
    ax.set_ylim(y_gnd - 0.6, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Tree (solid: wires, sources ○, loads □) and links (dotted: the cells)", loc="left", color=INK,
                 fontsize=10.5)

    ax = axes[1]
    ns = np.arange(1, 257)
    ax.plot(ns, [norm_lines(n) for n in ns], color=VIOLET, lw=2, label="line tree (cells are the links): closed form")
    ax.plot(ns, 2 * np.sqrt(2) / np.pi * (ns + 0.5), color=MUTED, lw=1.2, ls="--", label="0.90 (n + ½)")
    nd = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    dev = []
    for n in nd:
        cfg = Config(n, n, seed=0)
        dev.append(Interconnect(cfg, spanning_tree(cfg, "devices")).norm())
    ax.plot(nd, dev, color=INK_2, lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.5,
            label="device tree (bit wires and loads are the links): Lanczos")
    style(ax, "(b) ||M|| grows in proportion to the array's side", "array side n (n × n cells)", "||M||")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(nd, [str(n) for n in nd])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    save(fig, "loops.png")


def end_label(value):
    return f"{value:,.0f}" if value >= 100 else f"{value:.3g}"


def plot_series(ax, runs, sizes, solvers, value):
    """One line per solver, mean over converged seeds. Returns the end points to label (place_end_labels), leaving out
    one that would repeat a label already there (within 6%)."""
    ends = []
    for key, label, color, ls in solvers:
        pts = [(n, mean_over(runs[(key, n)], value)) for n in sizes if runs.get((key, n))]
        pts = [(n, y) for n, y in pts if y is not None]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, lw=2, ls=ls, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label, zorder=3)
        if all(abs(np.log(y[-1] / e[1])) > 0.06 for e in ends):
            ends.append((x[-1], y[-1]))
    return ends


def place_end_labels(ax, ends, gap=0.035):
    """Label each end point to its right, nudging labels apart vertically so that no two sit closer than gap (a fraction
    of the log axis's height). Call after the axis limits are final."""
    lo, hi = np.log10(ax.get_ylim())
    frac = lambda y: (np.log10(y) - lo) / (hi - lo)
    height = ax.get_window_extent().height * 72 / ax.figure.dpi  # axis height in points
    placed = []
    for x, y in sorted(ends, key=lambda e: e[1]):
        f = max(frac(y), placed[-1] + gap) if placed else frac(y)
        placed.append(f)
        ax.annotate(end_label(y), (x, y), xytext=(7, (f - frac(y)) * height), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8)


def scaling_figure():
    """Iterations, runtime and peak memory against size on the default crossbar."""
    runs = runner_runs()
    sizes = sorted({n for _, n in runs})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0), facecolor=SURFACE)
    panels = [("iterations (SPICE: Newton iterations)", iterations_of, "(a) Iterations to KCL tol 1e-9"),
              ("total runtime (s)", lambda r: r["runtime"], "(b) Runtime, laptop on AC (mean of 3 seeds)"),
              ("peak resident memory (GB)", lambda r: r["peak_memory"] / 1e9, "(c) Peak memory per solve")]
    ends = []
    for ax, (ylabel, value, title) in zip(axes, panels):
        ends.append(plot_series(ax, runs, sizes, SOLVERS, value))
        style(ax, title, "array size", ylabel, logy=True)
        size_axis(ax, sizes)
    # the rate predicted for the default rho (section 2.5): omega = 1 / (1 + tau R_min), iterations ln(1e9) / ln(1/omega)
    ns = np.array(sizes)
    omega = np.array([1 / (1 + np.sqrt(STEP_PRODUCT) / norm_lines(n)) for n in ns])
    axes[0].plot(ns, np.log(1e9) / np.log(1 / omega), color=VIOLET, lw=1.2, ls=":", zorder=2,
                 label="Condat–Vũ predicted: ln(10⁹) / ln(1/ω), ω = 1 / (1 + τ R_min)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ax, e in zip(axes, ends):
        place_end_labels(ax, e)
    fig.savefig(OUT / "scaling.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'scaling.png'}")


def tuning_records():
    """One tuning record per (size, rho): the "auto" runs count as their rho (1 kohm), which at 256x256 was only run as
    "auto"."""
    by = {}
    for r in records(RESULTS / "cv-tune" / "records.jsonl") + records(RESULTS / "cv-tune-256" / "records.jsonl"):
        by.setdefault((r["p"], round(r["rho"], 6)), r)
    return [dict(r, rho_option=r["rho"]) for r in by.values()]


def collapse_figure():
    """Every tuning run: (1 - observed rate) ||M|| against rho, which the bound predicts is size-independent."""
    recs = tuning_records()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=SURFACE)
    ax = axes[0]
    sizes = sorted({r["p"] for r in recs})
    shades = SIZE_RAMP[-len(sizes):]
    for n, shade in zip(sizes, shades):
        pts = sorted((r["rho"], r["iterations"]) for r in recs if r["p"] == n and r["rho_option"] != "auto")
        x, y = zip(*pts)
        ax.plot(x, y, color=shade, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2, label=f"{n}×{n}")
        best = min(pts, key=lambda t: t[1])
        ax.plot(*best, marker="o", ms=11, mfc="none", mec=INK, mew=1.3, zorder=5)
    ax.axvline(1000, color=MUTED, lw=1, ls="--")
    ax.text(1060, 0.03, "Chambolle–Pock balance\n√(R_min R_load) = 1 kΩ", transform=ax.get_xaxis_transform(),
            color=INK_2, fontsize=8.5)
    style(ax, "(a) Iterations against the step balance ρ (○ = best ρ at each size)", "ρ = √(σ / τ) (Ω)",
          "iterations to KCL tol 1e-9", logy=True)
    ax.set_xscale("log")
    ax.set_xticks(RHOS_TICKS, [f"{t:g}" for t in RHOS_TICKS])
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax = axes[1]
    rhos = np.geomspace(120, 3500, 300)
    bound = np.sqrt(STEP_PRODUCT) * np.minimum(1e3 / rhos, rhos / 1e3)
    ax.plot(rhos, bound, color=INK_2, lw=1.5, ls="--", label="Chambolle–Pock: √θ · min(R_min/ρ, ρ/R_load)")
    for n, shade in zip(sizes, shades):
        pts = sorted((r["rho"], (1 - r["observed_rate"]) * r["norm_M"]) for r in recs
                     if r["p"] == n and r["observed_rate"])
        x, y = zip(*pts)
        ax.plot(x, y, color=shade, lw=0, marker="o", ms=7, mec=SURFACE, mew=1.2)
    style(ax, "(b) (1 − rate) · ||M||: one curve for every size above the balance", "ρ = √(σ / τ) (Ω)",
          "(1 − observed rate) × ||M||")
    ax.set_xscale("log")
    ax.set_xticks(RHOS_TICKS, [f"{t:g}" for t in RHOS_TICKS])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    handles = [plt.Line2D([], [], color=shade, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.2) for shade in shades]
    handles.append(plt.Line2D([], [], color=INK_2, lw=1.5, ls="--"))
    labels = [f"{n}×{n}" for n in sizes] + ["Chambolle–Pock: √θ · min(R_min/ρ, ρ/R_load)"]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "rates.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'rates.png'}")


def convergence_figure(n=64, seed=0):
    """KCL residual against iteration at one size for every iterative solver, and Condat-Vu's fixed-point residual,
    the quantity the paper's stopping rule tests."""
    runs = runner_runs()
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=SURFACE)
    last = 1
    for key, label, color, ls in SOLVERS[:4]:
        rec = next(r for r in runs[(key, n)] if r["seed"] == seed)
        k = np.arange(1, rec["iterations"] + 1)
        last = max(last, k[-1])
        ax.plot(k, rec["history"]["kcl_history"], color=color, lw=2, ls=ls, label=f"{label}: KCL residual")
        if key.startswith("condat_vu") and ls == "-":
            ax.plot(k, rec["history"]["fixed_point_history"], color=color, lw=1.2, ls=":",
                    label=f"{label}: relative step ||z_k+1 − z_k|| / ||z_k+1|| (the paper's test)")
    ax.axhline(1e-9, color=MUTED, lw=1, ls="--")
    ax.text(0.99, 1.6e-9, "tol 1e-9", transform=ax.get_yaxis_transform(), ha="right", color=INK_2, fontsize=8.5)
    style(ax, f"Convergence on the {n}×{n} default crossbar (seed {seed})", "iteration (log scale)",
          "relative residual: KCL at the worst node (solid), step (dotted)", logy=True)
    ax.set_xscale("log")
    ax.set_xlim(0.9, last * 1.15)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    # ten decades: label every second one (1-2-5 sub-ticks leave matplotlib's locator with no ticks at all)
    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=7))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y >= 0.01 else f"1e{np.log10(y):.0f}"))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUT / "convergence.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'convergence.png'}")


def highR_figure():
    runs = runner_runs("highR")
    sizes = sorted({n for _, n in runs})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE)
    solvers = [(k, label.replace("√(R_min R_load)", "√(R_min R_load) = 10 kΩ"), c, ls) for k, label, c, ls in SOLVERS
               if k != "condat_vu_best"]
    ends = []
    for ax, (ylabel, value, title) in zip(axes, [
            ("iterations (SPICE: Newton iterations)", iterations_of, "(a) Iterations to KCL tol 1e-9"),
            ("total runtime (s)", lambda r: r["runtime"], "(b) Runtime, laptop on AC (mean of 3 seeds)")]):
        ends.append(plot_series(ax, runs, sizes, solvers, value))
        style(ax, title, "array size", ylabel, logy=True)
        size_axis(ax, sizes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("High-resistance crossbar: R_min 100 kΩ, R_max 10 MΩ, E 1–3 V", x=0.01, ha="left", color=INK,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ax, e in zip(axes, ends):
        place_end_labels(ax, e)
    fig.savefig(OUT / "highR.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'highR.png'}")


def s_min_lines(n):
    return lines_norm(n, n, smallest=True)


def tables():
    """The report's tables, as Markdown, from the same result files as the figures; printed and written to
    results/cv-tables.md."""
    lines = []
    emit = lines.append
    fmt = lambda x, d=3: "—" if x is None else f"{x:.{d}g}"

    emit("### ||M|| and its spread (line tree, closed form)\n")
    emit("| size | ||M|| | smallest singular value | ratio |")
    emit("|---|---|---|---|")
    for n in SIZES:
        emit(f"| {n}×{n} | {norm_lines(n):.2f} | {s_min_lines(n):.3f} | {norm_lines(n) / s_min_lines(n):.1f} |")

    for circuit, title in [("default", "default crossbar"), ("highR", "high-resistance crossbar")]:
        runs = runner_runs(circuit)
        keys = [s for s in SOLVERS if any(k == s[0] for k, _ in runs)]
        emit(f"\n### Runner sweep, {title}: iterations / runtime (s) / peak memory (GB), mean of converged seeds\n")
        emit("| size | " + " | ".join(label for _, label, _, _ in keys) + " |")
        emit("|---|" + "---|" * len(keys))
        for n in sorted({n for _, n in runs}):
            cells = []
            for key, *_ in keys:
                rs = runs.get((key, n), [])
                ok = [r for r in rs if r["status"] == "converged"]
                if not ok:
                    cells.append(f"{len(rs) - len(ok)} of {len(rs)} failed" if rs else "—")
                    continue
                its = [iterations_of(r) for r in ok]
                its_txt = f"{min(its)}" if min(its) == max(its) else f"{min(its)}–{max(its)}"
                cells.append(f"{its_txt} / {fmt(np.mean([r['runtime'] for r in ok]))} / "
                             f"{fmt(np.mean([r['peak_memory'] / 1e9 for r in ok]), 2)}")
            emit(f"| {n}×{n} | " + " | ".join(cells) + " |")

        emit(f"\n### Where the time goes, {title}, largest size (mean over seeds)\n")
        n = max(n for _, n in runs)
        emit("| solver | setup (s) | iterations | per iteration (ms) | iterations total (s) | runtime (s) |")
        emit("|---|---|---|---|---|---|")
        for key, label, *_ in keys:
            ok = [r for r in runs.get((key, n), []) if r["status"] == "converged"]
            if not ok:
                continue
            h = lambda name: float(np.mean([r["history"].get(name, np.nan) for r in ok]))
            its = float(np.mean([iterations_of(r) for r in ok]))
            if key == "spice":
                setup, loop = h("load_time") + h("netlist_time"), h("run_time")
            else:
                setup, loop = h("setup_time"), h("iteration_time")
            emit(f"| {label} | {setup:.3g} | {its:.0f} | {1e3 * loop / its:.3g} | {loop:.3g} | "
                 f"{np.mean([r['runtime'] for r in ok]):.3g} |")

    emit("\n### The paper's stopping rule: relative step when the KCL test passes, and where a step test would stop\n")
    runs = runner_runs()
    emit("| size | iterations (KCL test) | relative step then | stop at relative step ≤ 1e-9: iteration | "
         "KCL residual there |")
    emit("|---|---|---|---|---|")
    for n in SIZES:
        rec = next((r for r in runs.get(("condat_vu", n), []) if r["seed"] == 0), None)
        if rec is None:
            continue
        kcl, fp = np.array(rec["history"]["kcl_history"]), np.array(rec["history"]["fixed_point_history"])
        early = int(np.argmax(fp <= 1e-9)) if np.any(fp <= 1e-9) else None
        emit(f"| {n}×{n} | {rec['iterations']} | {fp[-1]:.1e} | "
             f"{'—' if early is None else early + 1} | {'—' if early is None else f'{kcl[early]:.1e}'} |")

    text = "\n".join(lines)
    print(text)
    (RESULTS / "cv-tables.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    loops_figure()
    scaling_figure()
    collapse_figure()
    convergence_figure()
    highR_figure()
    tables()
