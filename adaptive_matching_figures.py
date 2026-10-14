"""
Figures and summary tables for report_adaptive_matching.md, written to report_assets/adaptive_matching/ and stdout. Every
number comes from the runs the report describes:

    results/adaptive-1d1r                run_experiments.py (1D1R: fixed and adaptive matching, SPICE)
    results/adaptive-1s1r-Roff100000     run_experiments.py (1S1R, R_off 100 kohm: sector and adaptive matching, SPICE)
    results/adaptive-1s1r-Roff1000000    the same at R_off 1 Mohm
    results/adaptive-spectrum            adaptive_spectrum.py (1D1R: spectra at the final matching)

Accuracy is reported as each solver recorded it, by the same KCL test (algorithms/common.kcl_check) on its own node
potentials: the worst absolute residual and the worst relative one. On 1D1R arrays the relative residual is meaningless
on dead lines, whose nodes carry ~0 A, so the absolute one is the comparison there. Solutions are compared with SPICE's
through the saved pickles. (Projecting a Spingarn answer's v onto KVL by least squares instead, u = (A A^T)^-1 A v,
was tried and rejected: it spreads the KVL defect, <= tol max |v|, evenly over all branches, and 5-ohm wires turn it
into current errors that the solver's own, conductance-weighted, projection does not make.)

    python adaptive_matching_figures.py
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_assets" / "adaptive_matching"
SWEEP_1D1R = RESULTS / "adaptive-1d1r"
SWEEPS_1S1R = {"100 kΩ": RESULTS / "adaptive-1s1r-Roff100000", "1 MΩ": RESULTS / "adaptive-1s1r-Roff1000000"}
SPECTRUM = RESULTS / "adaptive-spectrum"

STYLE = {  # label, color, line style; colours as plotting.py assigns them
    "spingarn": ("Spingarn, fixed γ = R_min (α = 0.8)", SERIES_COLORS[0], "-"),
    "spingarn_sector": ("Spingarn, fixed sector matching (α = 1)", SERIES_COLORS[2], "-"),
    "spice": ("SPICE (Newton iterations)", SERIES_COLORS[1], "-"),
    "spingarn_adaptive": ("Spingarn, adaptive matching", SERIES_COLORS[6], "-"),
    "spingarn_adaptive_split": ("Spingarn, adaptive, split relaxation", SERIES_COLORS[6], "--"),
}


def load(run_dir):
    with open(Path(run_dir) / "runs.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_algorithm(runs):
    """{algorithm: {cells: [converged runs]}}, plus the failed runs separately."""
    ok, failed = defaultdict(lambda: defaultdict(list)), []
    for r in runs:
        (ok[r["algorithm"]][r["p"] * r["q"]] if r["status"] == "converged" else failed).append(r)
    return ok, failed


def mean(rs, key):
    return float(np.mean([key(r) for r in rs]))


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel("cells (p × q)", color=INK_2)
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


def plot_series(ax, ok, algorithms, value, label_last=None, min_gap=0.11):
    """One line per algorithm, mean over seeds; label_last labels each line's last point, with labels closer than
    min_gap decades pushed apart."""
    ends = []
    for alg in algorithms:
        if alg not in ok:
            continue
        label, color, ls = STYLE[alg]
        xs = sorted(ok[alg])
        ys = [mean(ok[alg][x], value) for x in xs]
        ax.plot(xs, ys, color=color, ls=ls, lw=2, marker="o", ms=8, mec=SURFACE, mew=1.5, label=label)
        if label_last:
            ends.append([np.log10(ys[-1]), xs[-1], ys[-1], label_last(ok[alg][xs[-1]])])
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + min_gap)
    for y_label, x, y, text in ends:
        ax.annotate(text, (x, y), xytext=(x * 1.25, 10 ** y_label), va="center", color=INK_2, fontsize=8.5)


def iterations_and_runtime(ok, algorithms, title, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    plot_series(axes[0], ok, algorithms, lambda r: r["iterations"])
    style(axes[0], f"{title}: iterations (SPICE: Newton iterations)", "iterations (mean of 3 seeds)")
    its = lambda rs: f"{np.mean([r['iterations'] for r in rs]):.0f} its"
    plot_series(axes[1], ok, algorithms, lambda r: r["runtime"], label_last=its)
    style(axes[1], f"{title}: total runtime", "seconds (mean of 3 seeds)")
    axes[1].set_xlim(None, axes[1].get_xlim()[1] * 2.5)  # room for the end labels
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=4)
    save(fig, name)


def residual_histories(runs, size, seed, name):
    """Fixed-point residual against iteration at one size and seed, each re-match marked on its curve."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for alg in ("spingarn", "spingarn_adaptive", "spingarn_adaptive_split"):
        r = next(r for r in runs if r["algorithm"] == alg and r["p"] == size and r["seed"] == seed)
        h = np.asarray(r["history"]["fixed_point_history"])
        label, color, ls = STYLE[alg]
        k = np.arange(1, h.size + 1)
        ax.plot(k, h, color=color, ls=ls, lw=2, label=f"{label}: {h.size} iterations")
        log = r["history"].get("rematch_log", [])
        if log:
            at = np.array([entry[0] for entry in log])
            ax.plot(at + 1, h[at], ls="none", marker="o", ms=7, mfc=SURFACE, mec=color, mew=1.8)
    ax.plot([], [], ls="none", marker="o", ms=7, mfc=SURFACE, mec=MUTED, mew=1.8, label="re-match (refactorization)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor(SURFACE)
    ax.set_title(f"1D1R {size}×{size}, seed {seed}: fixed-point residual per iteration", loc="left", color=INK,
                 fontsize=10.5)
    ax.set_xlabel("iteration", color=INK_2)
    ax.set_ylabel("relative fixed-point residual", color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=4)
    save(fig, name)


def accuracy(run_dir, runs, algorithms, reference="spice"):
    """Per algorithm and size, over seeds: max |v - v_reference| (from the pickles) and the worst recorded absolute
    and relative KCL residuals."""
    pickles = Path(run_dir) / "pickles"
    dv = defaultdict(float)
    for path in sorted(pickles.glob(f"{reference}_*.pkl")):
        with open(path, "rb") as f:
            ref = pickle.load(f)
        for alg in algorithms:
            other = path.with_name(path.name.replace(f"{reference}_", f"{alg}_", 1))
            if ref.converged and other.exists():
                with open(other, "rb") as f:
                    res = pickle.load(f)
                if res.converged:
                    dv[alg, ref.config.p] = max(dv[alg, ref.config.p], float(np.abs(res.v - ref.v).max()))
    groups = defaultdict(list)
    for r in runs:
        if r["status"] == "converged":
            groups[r["algorithm"], r["p"]].append(r["history"])
    for (alg, size), hs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if alg not in algorithms + [reference]:
            continue
        diff = f"max |dv| vs {reference} {dv[alg, size]:.1e} V, " if alg != reference else ""
        print(f"  {alg:<24} {size:>4}: {diff}KCL residual max {max(h['kcl_residual'] for h in hs):.1e} A, "
              f"relative {max(h['kcl_residual_rel'] for h in hs):.1e}")


def table(ok, failed, algorithms):
    """Markdown rows: runtime (iterations) per size, mean over seeds, with failures noted."""
    sizes = sorted({x for alg in algorithms for x in ok.get(alg, {})})
    print("| size | " + " | ".join(STYLE[a][0] for a in algorithms) + " |")
    print("|---" * (len(algorithms) + 1) + "|")
    for x in sizes:
        cells = []
        for alg in algorithms:
            rs = ok.get(alg, {}).get(x, [])
            n_fail = sum(1 for r in failed if r["algorithm"] == alg and r["p"] * r["q"] == x)
            if not rs:
                cells.append(f"fails ({n_fail})" if n_fail else "—")
                continue
            its = [r["iterations"] for r in rs]
            it = f"{min(its)}–{max(its)}" if min(its) != max(its) else f"{its[0]}"
            rm = [r["history"].get("rematches") for r in rs]
            extra = f", {min(rm)}–{max(rm)} re-matches" if rm[0] is not None else ""
            cells.append(f"{mean(rs, lambda r: r['runtime']):.3g} s ({it}{extra})" + (f", {n_fail} failed" if n_fail else ""))
        n = int(round(np.sqrt(x)))
        print(f"| {n}×{n} | " + " | ".join(cells) + " |")


def breakdown(ok, algorithms, size):
    """Where the time went at one size: setup (ordering + first factorization), re-match refactorizations, the rest of
    the iterations, and per-iteration cost, means over seeds."""
    for alg in algorithms:
        rs = ok.get(alg, {}).get(size * size, [])
        if not rs or alg == "spice":
            continue
        h = lambda key: mean(rs, lambda r: r["history"].get(key, 0.0) or 0.0)
        its = mean(rs, lambda r: r["iterations"])
        loop = h("iteration_time") - h("refactor_time")
        print(f"  {alg:<24} {size}: total {mean(rs, lambda r: r['runtime']):.3f} s = setup {h('setup_time'):.3f} "
              f"(factor {h('factor_time'):.3f}) + refactors {h('refactor_time'):.3f} + iterations {loop:.3f} "
              f"({1e3 * loop / its:.1f} ms each); peak memory {mean(rs, lambda r: r.get('peak_memory') or 0) / 1e9:.2f} GB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load(SWEEP_1D1R)
    ok, failed = by_algorithm(runs)
    algs = ["spingarn", "spice", "spingarn_adaptive", "spingarn_adaptive_split"]
    print("1D1R")
    table(ok, failed, algs)
    breakdown(ok, algs, 256)
    accuracy(SWEEP_1D1R, runs, ["spingarn", "spingarn_adaptive", "spingarn_adaptive_split"])
    iterations_and_runtime(ok, algs, "1D1R", "1d1r_scale.png")
    residual_histories(runs, 256, 0, "1d1r_residual_256.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    algs = ["spingarn_sector", "spice", "spingarn_adaptive"]
    its = lambda rs: f"{np.mean([r['iterations'] for r in rs]):.0f} its"
    for ax, (R_off, run_dir) in zip(axes, SWEEPS_1S1R.items()):
        print(f"1S1R, R_off {R_off}")
        sweep_runs = load(run_dir)
        sweep_ok, sweep_failed = by_algorithm(sweep_runs)
        table(sweep_ok, sweep_failed, algs)
        breakdown(sweep_ok, algs, 256)
        accuracy(run_dir, sweep_runs, ["spingarn_sector", "spingarn_adaptive"])
        plot_series(ax, sweep_ok, algs, lambda r: r["runtime"], label_last=its)
        style(ax, f"1S1R, R_off {R_off}: total runtime", "seconds (mean of 3 seeds)")
        ax.set_xlim(None, ax.get_xlim()[1] * 2.5)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=4)
    save(fig, "1s1r_scale.png")

    if (SPECTRUM / "records.jsonl").exists():
        print("spectra")
        for rec in load_jsonl(SPECTRUM / "records.jsonl"):
            f, a, s = rec["spingarn"], rec["spingarn_adaptive"], rec["spingarn_adaptive_split"]
            print(f"| {rec['p']}×{rec['q']} ({rec['seed']}) | {f['iterations']} | [{f['lam_min']:.4f}, {f['lam_max']:.3f}] "
                  f"| {a['iterations']} ({a['rematches']}) | [{a['lam_min']:.5f}, {a['lam_max']:.4f}] "
                  f"| {a['predicted_rate']:.3f} / {a['observed_rate']:.3f} | {s['iterations']} ({s['rematches']}) "
                  f"| {s['predicted_rate']:.3f} / {s['observed_rate']:.3f} | {a['open_matched']} / {a['reverse_at_solution']} |")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    main()
