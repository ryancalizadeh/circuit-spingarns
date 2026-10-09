"""
How Spingarn's factorization of A Gamma^-1 A^T scales: SuperLU vs CHOLMOD, each in its library's own fill-reducing
ordering (minimum degree / AMD) and in the geometric nested-dissection ordering (GNDO, algorithms/ordering.py), read from
experiment-runner results.

Prints markdown tables, per array size, of total runtime; of the factorization's cost (ordering + factorization time), its
fill (entries stored, and for CHOLMOD the exact nonzeros of L) and the process's peak memory; of what GNDO changes within
each library; of what a second BLAS thread changes for CHOLMOD; and of log-log scaling exponents against the node count
and against the stored entries. SPICE runs in the same directories (algorithm "spice") join the runtime, iteration,
accuracy and memory tables, with their own breakdown of where ngspice's time goes and how SPICE compares with Spingarn.
Draws the single-thread configurations, and SPICE where it has the metric, as small multiples to --plot.

    python compare_factorizations.py results/factorization results/slu_nd results/cholmod_2threads

Records are grouped by configuration (library, ordering, BLAS threads; SPICE by its matrix solver), read from each
directory's experiment.json and the runs' own history, so directories may be combined freely; only converged runs count
in the statistics, and every other outcome is listed.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plotting import GRID, INK, INK_2, MUTED, SURFACE, _color

ROOT = Path(__file__).resolve().parent
LIBRARIES = {"slu": "SuperLU", "cholmod": "CHOLMOD"}
ORDERINGS = {"mmd": "MMD", "amd": "AMD", "nested_dissection": "GNDO"}
# The single-thread configurations drawn, each in the color of the registered solver that runs it; GNDO on SuperLU is
# spingarn_slu with an option, so it shares that color and is told apart by its dashes and marker.
PLOTTED = [  # (label, color of, marker, line style)
    ("SuperLU · MMD", "spingarn_slu", "s", "-"),
    ("SuperLU · GNDO", "spingarn_slu", "D", (0, (4, 2))),
    ("CHOLMOD · AMD", "spingarn_cholmod", "^", "-"),
    ("CHOLMOD · GNDO", "spingarn", "o", "-"),
    ("SPICE · KLU", "spice", "v", (0, (1, 1.5))),
]


def _sum(*values):
    return None if any(v is None for v in values) else sum(values)


METRICS = {  # name -> value of one run record, or None where the run does not report it
    "runtime": lambda r: r["runtime"],
    "iterations": lambda r: r["iterations"],
    "kcl": lambda r: r["history"].get("kcl_residual_rel"),
    "factorization": lambda r: _sum(r["history"].get("ordering_time"), r["history"].get("factor_time")),
    "ordering": lambda r: r["history"].get("ordering_time"),
    "factor": lambda r: r["history"].get("factor_time"),
    "per_iteration": lambda r: None if "iteration_time" not in r["history"] else
    r["history"]["iteration_time"] / r["iterations"],
    "stored": lambda r: r["history"].get("factor_nnz"),
    "lnz": lambda r: r["history"].get("lnz"),
    "memory": lambda r: None if r.get("peak_memory") is None else r["peak_memory"] / 1e9,
    # SPICE only, from ngspice's own accounting (rusage) and the solver's timers
    "spice_load": lambda r: r["history"].get("load_time"),
    "spice_run": lambda r: r["history"].get("run_time"),
    "spice_reorder": lambda r: r["history"].get("matrix_reorder_time"),
    "spice_factor": lambda r: r["history"].get("matrix_factor_time"),
    "spice_lu": lambda r: _sum(r["history"].get("nonzeros"), r["history"].get("fill_in")),
    "spice_fill": lambda r: None if not r["history"].get("nonzeros") or r["history"].get("fill_in") is None else
    (r["history"]["nonzeros"] + r["history"]["fill_in"]) / r["history"]["nonzeros"],
}
SPICE = "SPICE · KLU"


def label(options, history):
    """Configuration of a run: library, ordering actually used, and BLAS threads when more than one."""
    text = f"{LIBRARIES[options['factorization']]} · {ORDERINGS.get(history['ordering_used'], history['ordering_used'])}"
    if options.get("gamma", "r_lo") != "r_lo":
        text += f" · gamma {options['gamma']}"
    threads = history.get("blas_threads")
    return text + (f" · {threads} threads" if threads not in (None, 1) else "")


def load(directories):
    """
    Spingarn and SPICE runs from every directory: the converged ones as {(configuration, n): [run record, ...]} for
    n x n arrays, and every other outcome as a list of (configuration, n, seed, status, error).
    """
    groups, failures = defaultdict(list), []
    for directory in directories:
        directory = Path(directory)
        options = {s["name"]: s for s in json.loads((directory / "experiment.json").read_text())["solvers"]}
        with open(directory / "runs.jsonl") as f:
            for line in f:
                r = json.loads(line) if line.strip() else None
                solver = options.get(r["algorithm"], {}) if r else {}
                if r and "factorization" in solver:
                    config = label(solver, r["history"]) if "ordering_used" in r["history"] else r["algorithm"]
                elif r and r["algorithm"] == "spice":
                    config = f"SPICE · {solver.get('matrix_solver', 'klu').upper()}"
                else:
                    continue
                if r["p"] != r["q"]:
                    raise ValueError(f"{directory}: only square arrays are compared, found {r['p']}x{r['q']}")
                if r["status"] == "converged":
                    groups[config, r["p"]].append(r)
                else:
                    failures.append((config, r["p"], r["seed"], r["status"], r.get("error")))
    return groups, failures


def aggregate(groups):
    """{configuration: {n: {metric: (mean, std, runs, max)}}} over each group's runs that report the metric."""
    table = defaultdict(dict)
    for (config, n), runs in groups.items():
        stats = {}
        for name, value in METRICS.items():
            values = np.array([v for v in map(value, runs) if v is not None], dtype=float)
            stats[name] = (values.mean(), values.std(), len(values), values.max()) if values.size else None
        table[config][n] = stats
    return table


WORST = {"kcl"}  # metrics tabulated by their worst (largest) run rather than the mean


def cell(table, config, n, metric, fmt, scale=1.0):
    stats = table.get(config, {}).get(n, {}).get(metric)
    return "—" if stats is None else fmt.format(stats[3 if metric in WORST else 0] * scale)


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def ratio(table, before, after, n, metric):
    a, b = (table.get(c, {}).get(n, {}).get(metric) for c in (before, after))
    return "—" if a is None or b is None or b[0] == 0 else f"{a[0] / b[0]:.2f}×"


def exponent(table, config, x_metric, y_metric, min_n):
    """Least-squares slope of log y against log x (x_metric "nodes" is the node count 2 n^2) over sizes >= min_n."""
    points = []
    for n, stats in table.get(config, {}).items():
        x = 2 * n * n if x_metric == "nodes" else (stats[x_metric] or (None,))[0]
        y = (stats[y_metric] or (None,))[0]
        if n >= min_n and x and y:
            points.append((np.log(x), np.log(y)))
    return f"{np.polyfit(*zip(*points), 1)[0]:.2f}" if len(points) >= 3 else "—"


def report(table, min_n, failures=()):
    configs = [c for c in [p[0] for p in PLOTTED] + sorted(table) if c in table]
    configs = list(dict.fromkeys(configs))  # plotted order first, then any others
    sizes = sorted({n for c in configs for n in table[c]})
    out = []

    out.append("**Total runtime (s)**, mean (± std) over converged runs\n")
    out.append(markdown(["size"] + configs, [[f"{n}×{n}"] + [
        "—" if table[c].get(n, {}).get("runtime") is None else
        "{:.3g} (±{:.2g})".format(*table[c][n]["runtime"][:2]) for c in configs] for n in sizes]))

    if failures:
        out.append("\n**Runs that did not converge** (left out of every statistic)\n")
        out.append(markdown(["configuration", "size", "seed", "status", "error"],
                            [[c, f"{n}×{n}", str(seed), status, error or ""]
                             for c, n, seed, status, error in sorted(failures, key=lambda f: (f[0], f[1], f[2]))]))

    for metric, title, fmt, scale in [("iterations", "Iterations (Spingarn: linear solves; SPICE: Newton)", "{:.3g}",
                                       1),
                                      ("kcl", "KCL residual of the answer, relative (max over converged runs)",
                                       "{:.1e}", 1),
                                      ("factorization", "Ordering + factorization (s)", "{:.3g}", 1),
                                      ("per_iteration", "Time per iteration (ms)", "{:.3g}", 1e3),
                                      ("stored", "Factor entries stored (millions; L+U for SuperLU, L for CHOLMOD)",
                                       "{:.3g}", 1e-6),
                                      ("lnz", "Exact nonzeros of L (millions, CHOLMOD)", "{:.3g}", 1e-6),
                                      ("memory", "Peak resident memory of the run's process (GB)", "{:.3g}", 1)]:
        cols = [c for c in configs if any(table[c].get(n, {}).get(metric) for n in sizes)]
        out.append(f"\n**{title}**\n")
        out.append(markdown(["size"] + cols, [[f"{n}×{n}"] + [cell(table, c, n, metric, fmt, scale) for c in cols]
                                              for n in sizes]))

    pairs = [("CHOLMOD · AMD", "CHOLMOD · GNDO"), ("SuperLU · MMD", "SuperLU · GNDO"), ("SuperLU · MMD", "CHOLMOD · GNDO")]
    for before, after in pairs:
        if before in table and after in table:
            out.append(f"\n**{before} ÷ {after}** (above 1: {after} is smaller or faster)\n")
            out.append(markdown(["size", "runtime", "ordering + factorization", "stored entries", "nonzeros of L",
                                 "peak memory"],
                                [[f"{n}×{n}"] + [ratio(table, before, after, n, m) for m in
                                                 ("runtime", "factorization", "stored", "lnz", "memory")]
                                 for n in sizes]))

    threaded = sorted(c for c in table if c.endswith("threads"))
    if threaded:
        out.append("\n**BLAS threads (CHOLMOD): 1 thread ÷ more threads** (above 1: the threads help)\n")
        rows = []
        for c in threaded:
            single = c.rsplit(" · ", 1)[0]
            for n in sizes:
                if n in table[c] and n in table.get(single, {}):
                    rows.append([c, f"{n}×{n}"] + [ratio(table, single, c, n, m) for m in ("runtime", "factor")])
        out.append(markdown(["configuration", "size", "runtime", "factorization"], rows))

    if SPICE in table:
        spice_sizes = sorted(table[SPICE])
        out.append(f"\n**Where {SPICE}'s time goes** (mean; load = netlist parsing and setup, run = the .op analysis; "
                   "reorder and factor are ngspice's own totals over all Newton iterations)\n")
        out.append(markdown(["size", "total (s)", "load (s)", "run (s)", "reorder (s)", "factor (s)",
                             "L+U entries (millions)", "L+U ÷ matrix nonzeros"],
                            [[f"{n}×{n}"] + [cell(table, SPICE, n, m, fmt, scale) for m, fmt, scale in
                                             [("runtime", "{:.3g}", 1), ("spice_load", "{:.3g}", 1),
                                              ("spice_run", "{:.3g}", 1), ("spice_reorder", "{:.3g}", 1),
                                              ("spice_factor", "{:.3g}", 1), ("spice_lu", "{:.3g}", 1e-6),
                                              ("spice_fill", "{:.3g}", 1)]] for n in spice_sizes]))
        others = [c for c in configs if c != SPICE and not c.endswith("threads")]
        out.append(f"\n**{SPICE} ÷ Spingarn** (above 1: Spingarn is faster or smaller)\n")
        out.append(markdown(["size"] + [f"runtime ÷ {c}" for c in others] + [f"memory ÷ {c}" for c in others],
                            [[f"{n}×{n}"] + [ratio(table, SPICE, c, n, "runtime") for c in others] +
                             [ratio(table, SPICE, c, n, "memory") for c in others] for n in spice_sizes]))

    out.append(f"\n**Scaling exponents** (log-log least-squares slope over sizes ≥ {min_n}×{min_n}; nodes = 2n²)\n")
    out.append(markdown(["configuration", "runtime ∝ nodes^", "ordering + factorization ∝ nodes^", "stored ∝ nodes^",
                         "ordering + factorization ∝ stored^", "nonzeros of L ∝ nodes^", "peak memory ∝ nodes^"],
                        [[c] + [exponent(table, c, x, y, min_n) for x, y in [("nodes", "runtime"),
                                                                             ("nodes", "factorization"),
                                                                             ("nodes", "stored"),
                                                                             ("stored", "factorization"),
                                                                             ("nodes", "lnz"),
                                                                             ("nodes", "memory")]]
                         for c in configs]))
    return "\n".join(out)


def _end_labels(ax, ends, min_gap=10.5):
    """
    Label each line at its right end, in ink rather than the series color. Labels closer than min_gap points are pushed
    apart symmetrically about where their lines end, and a displaced label gets a thin connector to its line's end.
    """
    if not ends:
        return
    points_per_pixel = 72 / ax.figure.dpi
    ends = sorted((ax.transData.transform((x, y))[1] * points_per_pixel, x, y, text) for x, y, text in ends)
    placed = [e[0] for e in ends]
    for _ in range(100):  # relax overlapping neighbours apart until every gap is wide enough
        moved = False
        for i in range(1, len(placed)):
            overlap = min_gap - (placed[i] - placed[i - 1])
            if overlap > 1e-6:
                placed[i - 1] -= overlap / 2
                placed[i] += overlap / 2
                moved = True
        if not moved:
            break
    for (y_pt, x, y, text), y_new in zip(ends, placed):
        shift = y_new - y_pt
        connector = dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=1, shrinkB=5) if abs(shift) > 2 else None
        ax.annotate(text, (x, y), xytext=(14, shift), textcoords="offset points", va="center", color=INK_2,
                    fontsize=8, annotation_clip=False, arrowprops=connector)


def plot(table, path, failures=()):
    """Small multiples of the PLOTTED configurations. A size at which every run of a plotted configuration failed is
    marked with a cross at the bottom of the runtime panel, in that configuration's color."""
    panels = [("runtime", "Total runtime", "seconds", 1),
              ("factorization", "Ordering + factorization", "seconds", 1),
              ("stored", "Factor entries stored", "millions", 1e-6),
              ("memory", "Peak resident memory of the run", "GB", 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=SURFACE)
    plotted = {p[0] for p in PLOTTED}
    failed = sorted({(c, n) for c, n, *_ in failures if c in plotted and n not in table.get(c, {})})
    sizes = sorted({n for c in table for n in table[c]} | {n for _, n in failed})
    handles = {}
    for ax, (metric, title, unit, scale) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ends = []
        for config, color_of, marker, style in PLOTTED:
            pts = sorted((n, s[metric]) for n, s in table.get(config, {}).items() if s.get(metric))
            if not pts:
                continue
            x = np.array([2 * n * n for n, _ in pts], dtype=float)
            y = np.array([s[0] for _, s in pts]) * scale
            (line,) = ax.plot(x, y, color=_color(color_of, []), linestyle=style, linewidth=2, marker=marker,
                              markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=config)
            handles[config] = line
            ends.append((x[-1], y[-1], config.replace("·", "").replace("  ", " ")))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [2 * n * n for n in sizes]
        ax.set_xticks(ticks, [str(n) if n & (n - 1) == 0 else "" for n in sizes])  # other sizes would crowd them
        if metric == "runtime":
            for config, n in failed:
                color_of = next(p[1] for p in PLOTTED if p[0] == config)
                top = ax.get_xaxis_transform()  # x in data, y in axes fraction
                ax.plot([2 * n * n], [0.05], transform=top, marker="X", markersize=9, color=_color(color_of, []),
                        markeredgecolor=SURFACE, clip_on=False, linestyle="none")
                ax.annotate(f"{config.split(' · ')[0]} fails at {n}×{n}", (2 * n * n, 0.05), xycoords=top,
                            xytext=(9, 0), textcoords="offset points", va="center", color=INK_2, fontsize=8)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))  # labels even on a panel spanning < 1 decade
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlim(ticks[0] / 1.6, ticks[-1] * 9)  # room on the right for the end labels
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylabel(unit, color=INK_2)
        ax.set_xlabel("array size n (n×n cross-points)", color=INK_2)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
        fig.canvas.draw()  # fixes the transforms that place the end labels
        _end_labels(ax, ends)
    fig.legend([handles[c] for c, *_ in PLOTTED if c in handles], [c for c, *_ in PLOTTED if c in handles],
               loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=len(handles), frameon=False, labelcolor=INK_2,
               handlelength=2.6)
    fig.text(0.01, 0.005, "mean over converged runs, 1 BLAS thread; SPICE (ngspice, KLU) has no single factorization to "
             "plot, so it appears in the runtime and memory panels only; log-log axes", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="+", type=Path, help="experiment-runner output directories")
    ap.add_argument("--plot", type=Path, default=ROOT / "report_assets" / "factorization_comparison.png")
    ap.add_argument("--min-size", type=int, default=128, help="smallest n used for the scaling exponents")
    ap.add_argument("--markdown", type=Path, help="also write the tables to this file")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the tables use ×, ÷, ≥ and ·, which Windows' console code page lacks
    groups, failures = load(args.directories)
    table = aggregate(groups)
    text = report(table, args.min_size, failures)
    print(text)
    if args.markdown:
        args.markdown.write_text(text + "\n", encoding="utf-8")
    plot(table, args.plot, failures)
    print(f"\nplot saved to {args.plot}")


if __name__ == "__main__":
    main()
