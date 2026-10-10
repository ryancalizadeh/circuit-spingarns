"""
Feasibility check for the 1S1R crossbar (a threshold selector in series with each memristor, see devices.py): does
SPICE's Newton iteration struggle as the selector's knee sharpens, while Spingarn's iteration count stays put?

For each selector off-resistance R_off and knee half-width delta (delta = 0: piecewise linear, a kink at |v| = V_th),
and each size and seed, one circuit is solved by
    SPICE            ngspice, KLU, RELTOL 1e-3: its Newton iterations and how it reached the operating point;
    spingarn_sector  Spingarn with each cell matched over its sector [R_on + R_min, R_off + R_max], alpha = 1: an a priori
                     contraction bound that does not depend on delta;
    spingarn         Spingarn with each cell matched to its on state R_on + R_min, alpha = 1.
Also recorded: agreement between the solvers, and where the selectors sit at the solution (conducting, or within
10 mV of the threshold).

Solves run in this process: the circuits are small and runtimes are not the point.

    python selector_feasibility.py [--sizes 1 16] [--seeds 3]
    python selector_feasibility.py --replot results/selector-feasibility-20261009-180000
"""
import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Config import BranchKind, Config
from algorithms.spice import SpiceSolver
from algorithms.spingarns import SpingarnSectorSolver, SpingarnSolver
from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE
from run_experiments import git_commit, parse_size, run_one

ROOT = Path(__file__).resolve().parent
CIRCUIT = dict(device="1s1r", E_range=(-1.0, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5), R_on=1e3, V_th=0.5)
R_OFFS = (1e5, 1e6)
DELTAS = (0.1, 0.03, 0.01, 0.003, 0.001, 0.0)
SOLVERS = ("spice", "spingarn_sector", "spingarn")


def selectors(cfg, v):
    """Where the selectors sit at the solution v."""
    _, _, v_s, v_m, _ = cfg.series_solve(v[cfg.slices[BranchKind.DEVICE]])
    on = np.abs(v_s) > cfg.V_th
    return dict(on=float(np.mean(on)), near_threshold=float(np.mean(np.abs(np.abs(v_s) - cfg.V_th) <= 0.01)),
                in_knee=float(np.mean(np.abs(np.abs(v_s) - cfg.V_th) < cfg.delta)),
                v_mem_on_median=float(np.median(np.abs(v_m[on]))) if on.any() else None)


def run_case(solvers, p, q, seed, R_off, delta):
    cfg = Config(p, q, seed=seed, R_off=R_off, delta=delta, **CIRCUIT)
    d = cfg.slices[BranchKind.DEVICE]
    K = cfg.r_hi[d] / cfg.r_lo[d]
    record = dict(R_off=R_off, delta=delta, p=p, q=q, seed=seed, K_min=float(K.min()), K_max=float(K.max()), solves={})
    solutions = {}
    for solver in solvers:
        res = run_one(solver, cfg)
        h = res.history
        entry = dict(status=res.status, iterations=None if np.isnan(res.iterations) else int(res.iterations),
                     runtime=res.runtime, kcl_residual_rel=h.get("kcl_residual_rel"), error=res.error)
        if solver.name == "spice":
            entry.update(method=h.get("method"), newton_iterations=h.get("spice_iterations"), notes=h.get("notes", [])[:4])
        else:
            entry.update(contraction_bound=h.get("contraction_bound"))
        record["solves"][solver.name] = entry
        if res.status == "converged":
            solutions[solver.name] = res.v
    for name in ("spingarn_sector", "spingarn"):
        if name in solutions and "spice" in solutions:
            record["solves"][name]["max_dv_vs_spice"] = float(np.abs(solutions[name] - solutions["spice"]).max())
    if solutions:
        record["selectors"] = selectors(cfg, next(iter(solutions.values())))
    return record


def report(rec):
    s = rec["solves"]
    sp = s["spice"]
    spice = f"{sp['newton_iterations']:.0f} its, {sp['method']}" if sp["status"] == "converged" else sp["status"]
    parts = [f"R_off {rec['R_off']:.0e} delta {rec['delta'] * 1e3:5.1f} mV {rec['p']}x{rec['q']} seed {rec['seed']}:",
             f"spice {spice}"]
    for name in ("spingarn_sector", "spingarn"):
        e = s[name]
        its = e["iterations"] if e["status"] == "converged" else f"{e['status']}"
        dv = f", dv {e['max_dv_vs_spice']:.0e}" if "max_dv_vs_spice" in e else ""
        parts.append(f"{name} {its} (bound {e['contraction_bound']:.3f}{dv})")
    sel = rec.get("selectors", {})
    parts.append(f"on {sel.get('on', float('nan')):.2f}, near V_th {sel.get('near_threshold', float('nan')):.3f}")
    print(" | ".join(parts), flush=True)


def load(out_dir):
    with open(Path(out_dir) / "records.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def plot(out_dir):
    records = load(out_dir)
    spec = json.loads((Path(out_dir) / "experiment.json").read_text())
    p, q = max((r["p"], r["q"]) for r in records)
    deltas = spec["deltas"]
    x = np.arange(len(deltas))
    labels = [f"{d * 1e3:g}" if d > 0 else "0\n(kink)" for d in deltas]
    series = [("spice", "SPICE: Newton iterations", SERIES_COLORS[1], "newton_iterations"),
              ("spingarn_sector", "Spingarn, matched over the sector", SERIES_COLORS[2], "iterations"),
              ("spingarn", "Spingarn, matched to the on state", SERIES_COLORS[0], "iterations")]
    fig, axes = plt.subplots(1, len(spec["R_offs"]), figsize=(13, 5), facecolor=SURFACE, sharey=True)
    for ax, R_off in zip(np.atleast_1d(axes), spec["R_offs"]):
        rows = [r for r in records if r["R_off"] == R_off and (r["p"], r["q"]) == (p, q)]
        for name, label, color, key in series:
            means, fallback = [], []
            for delta in deltas:
                vals = [r["solves"][name][key] for r in rows if r["delta"] == delta
                        and r["solves"][name]["status"] == "converged"]
                means.append(np.mean(vals) if vals else np.nan)
                fallback.append(any(r["solves"][name].get("method") not in (None, "newton")
                                    for r in rows if r["delta"] == delta))
            ax.plot(x, means, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=1.5, label=label)
            for k in np.flatnonzero(fallback):
                ax.plot(x[k], means[k], "o", ms=13, mfc="none", mec=INK_2, mew=1.2)
        K = [r["K_max"] for r in rows]
        bound = rows[0]["solves"]["spingarn_sector"]["contraction_bound"]
        ax.set_title(f"R_off = {R_off / 1e3:g} kΩ  (cell K up to {max(K):.0f}; sector-matched bound {bound:.3f})",
                     loc="left", color=INK, fontsize=10.5)
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_xticks(x, labels)
        ax.set_xlabel("selector knee half-width δ (mV)", color=INK_2)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel(f"iterations, {p}×{q}, mean of {spec['seeds']} seeds", color=INK_2)
    first.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    fallbacks = sum(r["solves"]["spice"].get("method") not in (None, "newton") for r in records)
    fig.suptitle("1S1R crossbar: does a sharper selector knee hurt SPICE's Newton iteration?\n"
                 f"V_th = {CIRCUIT['V_th']} V, R_on = {CIRCUIT['R_on'] / 1e3:g} kΩ, E ∈ [{CIRCUIT['E_range'][0]:g}, "
                 f"{CIRCUIT['E_range'][1]:g}] V. " + (f"Circled: SPICE needed a fallback ({fallbacks} runs)."
                 if fallbacks else "SPICE converged by plain Newton in every run.") + f" Measured on {spec['machine']}.",
                 x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    path = Path(out_dir) / "selector_feasibility.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=[(1, 1), (16, 16)])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--replot", type=Path, metavar="DIR")
    args = ap.parse_args()
    if args.replot:
        print(f"wrote {plot(args.replot)}")
        return
    out_dir = args.out or ROOT / "results" / datetime.now().strftime("selector-feasibility-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "experiment.json").write_text(json.dumps(dict(
        started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(), sizes=args.sizes,
        seeds=args.seeds, R_offs=R_OFFS, deltas=DELTAS, circuit=CIRCUIT, solvers=SOLVERS, git_commit=git_commit()),
        indent=2))
    solvers = [SpiceSolver(), SpingarnSectorSolver(), SpingarnSolver()]
    with open(out_dir / "records.jsonl", "a") as log:
        for R_off in R_OFFS:
            for delta in DELTAS:
                for p, q in args.sizes:
                    for seed in range(args.seeds):
                        rec = run_case(solvers, p, q, seed, R_off, delta)
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        report(rec)
    print(f"wrote {plot(out_dir)}")
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
