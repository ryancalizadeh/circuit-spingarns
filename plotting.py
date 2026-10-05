import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from algorithms import SOLVERS

# Categorical slots in fixed order, assigned by registry order so an algorithm keeps its color whatever subset
# of algorithms a sweep runs.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"


def _color(alg, algorithms):
    order = list(SOLVERS) + [a for a in algorithms if a not in SOLVERS]
    return SERIES_COLORS[order.index(alg) % len(SERIES_COLORS)]


def load_runs(run_dir):
    with open(Path(run_dir) / "runs.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(runs):
    """
    Aggregate run records per (algorithm, p, q). Runtime and iterations are averaged over converged runs only:
    a run that hit its iteration budget or raised has no meaningful runtime to compare.
    """
    groups = defaultdict(list)
    for r in runs:
        groups[(r["algorithm"], r["p"], r["q"])].append(r)
    rows = []
    for (alg, p, q), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] * kv[0][2], kv[0][1])):
        ok = [r for r in rs if r["status"] == "converged"]
        rt = np.array([r["runtime"] for r in ok], dtype=float)
        it = np.array([r["iterations"] for r in ok], dtype=float)
        rows.append(dict(
            algorithm=alg, p=p, q=q, cells=p * q, branches=3 * p * q,
            runs=len(rs), converged=len(ok),
            not_converged=sum(r["status"] == "not_converged" for r in rs),
            errors=sum(r["status"] == "error" for r in rs),
            crashed=sum(r["status"] == "crashed" for r in rs),
            timeouts=sum(r["status"] == "timeout" for r in rs),
            runtime_mean=float(rt.mean()) if ok else None, runtime_std=float(rt.std()) if ok else None,
            iterations_mean=float(it.mean()) if ok else None, iterations_std=float(it.std()) if ok else None,
        ))
    return rows


def _style(ax, rows, title, ylabel, logy=True):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=30)
    ax.set_xlabel("array size p×q (log scale in cross-points)", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    # One x tick per swept size, labeled with its dimensions.
    ax.set_xscale("log")
    labels = defaultdict(set)
    for r in rows:
        labels[r["cells"]].add(f"{r['p']}×{r['q']}")
    ax.set_xticks(sorted(labels), ["\n".join(sorted(labels[c])) for c in sorted(labels)])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
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


def _legend(ax):
    """Legend in one row between the title and the plot, clear of the data."""
    ax.legend(frameon=False, labelcolor=INK_2, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=max(1, len(ax.get_legend_handles_labels()[0])), borderaxespad=0.3, handlelength=1.8)


def _plot_metric(rows, algorithms, key, title, ylabel, path, logy=True):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style(ax, rows, title, ylabel, logy)
    drawn = 0
    for alg in algorithms:
        pts = [r for r in rows if r["algorithm"] == alg and r[f"{key}_mean"] is not None]
        if not pts:
            continue
        x = np.array([r["cells"] for r in pts])
        y = np.array([r[f"{key}_mean"] for r in pts])
        s = np.array([r[f"{key}_std"] for r in pts])
        color = _color(alg, algorithms)
        ax.fill_between(x, np.maximum(y - s, y * 1e-3), y + s, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=alg)
        ax.annotate(alg, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
        drawn += 1
    if drawn:
        _legend(ax)
    else:
        ax.text(0.5, 0.5, "no converged runs", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    fig.text(0.01, 0.01, "mean ± std over converged runs", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plot_convergence(rows, algorithms, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style(ax, rows, "Fraction of runs converged", "converged / runs", logy=False)
    ax.set_ylim(-0.05, 1.05)
    for alg in algorithms:
        pts = [r for r in rows if r["algorithm"] == alg]
        if not pts:
            continue
        x = np.array([r["cells"] for r in pts])
        y = np.array([r["converged"] / r["runs"] for r in pts])
        color = _color(alg, algorithms)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=alg)
    if rows:
        _legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_plots(run_dir, algorithms=None):
    """
    Aggregate `runs.jsonl` in `run_dir`, write `summary.json`, and draw runtime.png, iterations.png and
    convergence.png. `algorithms` fixes the color order; it defaults to the order recorded in experiment.json.
    """
    run_dir = Path(run_dir)
    if algorithms is None:
        with open(run_dir / "experiment.json") as f:
            algorithms = [s["name"] for s in json.load(f)["solvers"]]
    rows = summarize(load_runs(run_dir))
    with open(run_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    _plot_metric(rows, algorithms, "runtime", "Runtime vs array size", "runtime (s)", run_dir / "runtime.png")
    _plot_metric(rows, algorithms, "iterations", "Iteration count vs array size", "iterations",
                 run_dir / "iterations.png")
    _plot_convergence(rows, algorithms, run_dir / "convergence.png")
    return rows
