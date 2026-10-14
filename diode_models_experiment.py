"""
Diode models in the 1D1R crossbar: how the diode law (Shockley or ideal) and the formulation (lumped cell or diode on a
branch of its own, through an internal node) change Newton (SPICE and in-repo) against Spingarn. Three cases:

    lumped    a Shockley diode in series with each memristor, one branch per cell (device "1d1r_shockley");
    internal  the same circuit with each diode on its own branch, through an internal node (internal_node=True);
    ideal     the ideal diode on its own branch (device "1d1r", internal_node=True), same circuit as report_nonsmooth's 1D1R.
The scaling sweep adds
    ideal_lumped  report_nonsmooth's 1D1R itself (device "1d1r"), for the Newton baseline on Spingarn's own factorization
                  that report_adaptive_matching.md section 4 asks for.

The circuit is the 1D1R circuit of report_nonsmooth.md section 2 (E in [-1, 1.5] V, R_min 1-10 kohm, R_max 50-200 kohm
per cell, wires 5 ohm, sources 50 ohm, loads 1 kohm); the diode's I_s and n vary.

feasibility  in-process, small sizes: every solver variant of every case, agreement with Newton on the lumped cell at the
             word and bit nodes, SPICE's method, and where the diodes operate. Runtimes are single in-process timings and
             are recorded for orientation only.
scale        the experiment runner's isolated runs (run_experiments.run_isolated), every case's chosen solvers interleaved
             per size and seed in one sweep, with a timeout; adds the case and diode parameters to each run record.

Operating point, from Newton's solution of the lumped cell: a diode is reverse-biased if v_d < 0; a forward-biased diode
is subthreshold if its incremental resistance r_d = n V_T / (i + I_s) exceeds 10 R_min of its cell, on if r_d < R_min /
10, and in the knee in between, where diode and memristor share the cell's voltage and its law is curved. The output
change is the relative change of the bit-line output currents (through the loads) from the ideal-diode 1D1R crossbar.

    python diode_models_experiment.py feasibility [--sizes 1 2 4 8 16 32] [--seeds 3]
    python diode_models_experiment.py scale --sizes 8 16 32 64 128 256 --seeds 3 --timeout 1800 --diode 1e-6 1.0
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from Config import BranchKind, Config
from algorithms.newton import NewtonSafeguardedSolver, NewtonSolver
from algorithms.spice import SpiceExpSolver, SpiceSolver
from algorithms.spingarns import SpingarnAdaptiveSolver, SpingarnAdaptiveSplitSolver, SpingarnSolver
from run_experiments import git_commit, parse_size, run_isolated, run_one

ROOT = Path(__file__).resolve().parent
CIRCUIT = dict(E_range=(-1.0, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
DIODES = [(1e-14, 1.0), (1e-9, 1.0), (1e-6, 1.0), (1e-9, 1.5)]
CASES = ("lumped", "internal", "ideal")
SCALE_CASES = CASES + ("ideal_lumped",)


def circuit(case, I_s, n):
    """Config keyword arguments of a case."""
    if case == "ideal":
        return dict(CIRCUIT, device="1d1r", internal_node=True)
    if case == "ideal_lumped":
        return dict(CIRCUIT, device="1d1r")
    return dict(CIRCUIT, device="1d1r_shockley", I_s=I_s, n=n, internal_node=case == "internal")


def named(solver, name):
    """solver under another name, so several settings of one class can share a sweep."""
    solver.name = name
    return solver


def feasibility_solvers(case):
    if case == "lumped":
        return [NewtonSolver(), NewtonSafeguardedSolver(), named(SpingarnSolver(alpha=0.8, max_iterations=5000),
                                                                 "spingarn_fixed"),
                SpingarnAdaptiveSolver(), SpingarnAdaptiveSplitSolver(max_iterations=3000)]
    if case == "internal":
        return [NewtonSolver(), NewtonSafeguardedSolver(), SpiceSolver(), SpiceExpSolver(),
                named(SpingarnSolver(alpha=0.8, max_iterations=5000), "spingarn_fixed"), SpingarnAdaptiveSolver(),
                SpingarnAdaptiveSplitSolver(max_iterations=3000)]
    solvers = [named(SpiceSolver(ron=ron), f"spice_ron{ron:g}") for ron in (1.0, 1e-3, 1e-6)]
    for alpha in (0.5, 2 / 3, 0.8):
        for gamma, label in ((100.0, "100"), ("r_min", "rmin"), (1e5, "1e5")):
            solvers.append(named(SpingarnSolver(alpha=alpha, diode_gamma=gamma, max_iterations=3000),
                                 f"spingarn_a{alpha:.2f}_g{label}"))
    solvers.append(SpingarnAdaptiveSolver())
    return solvers


def scale_solvers(case):
    """The solvers of the scaling sweep, chosen after the feasibility runs: spingarn_adaptive_split is left out of the
    Shockley cases, where it failed to converge on most circuits from 32x32 up (it stalls on strongly reflecting
    unopened branches), and of the ideal internal-node case, where a conducting diode reflects -1 against any port."""
    if case == "lumped":
        return [NewtonSolver(), NewtonSafeguardedSolver(), named(SpingarnSolver(alpha=0.8), "spingarn_fixed"),
                SpingarnAdaptiveSolver()]
    if case == "internal":
        return [NewtonSolver(), NewtonSafeguardedSolver(), SpiceSolver(), SpiceExpSolver(),
                named(SpingarnSolver(alpha=0.8), "spingarn_fixed"), SpingarnAdaptiveSolver()]
    if case == "ideal_lumped":
        return [NewtonSolver(), NewtonSafeguardedSolver(), SpiceSolver(), SpingarnAdaptiveSolver(),
                SpingarnAdaptiveSplitSolver()]
    return [named(SpiceSolver(ron=1e-3), "spice_ron0.001"),
            named(SpingarnSolver(alpha=0.8, diode_gamma="r_min"), "spingarn_fixed"), SpingarnAdaptiveSolver()]


def operating_point(cfg, v):
    """Where the diodes of the lumped Shockley crossbar cfg sit at its solution v (see the module docstring)."""
    d = cfg.slices[BranchKind.DEVICE]
    _, i, v_d, v_m, _ = cfg.diode_series_solve(v[d])
    with np.errstate(divide="ignore"):  # i = -I_s to rounding in deep reverse bias: r_d = inf
        r_d = cfg.n * cfg.V_T / (i + cfg.I_s)
    fwd = v_d >= 0
    R = cfg.R_min_cell
    q = lambda arr, p: float(np.quantile(arr, p)) if arr.size else None
    return dict(reverse=float(np.mean(~fwd)), subthreshold=float(np.mean(fwd & (r_d > 10 * R))),
                knee=float(np.mean(fwd & (r_d <= 10 * R) & (r_d >= R / 10))), on=float(np.mean(fwd & (r_d < R / 10))),
                v_diode_fwd_median=q(v_d[fwd], 0.5), v_cell_fwd_median=q(v[d][fwd], 0.5),
                v_mem_fwd_median=q(v_m[fwd], 0.5), i_fwd_median=q(i[fwd], 0.5))


def output_change(cfg, i_ref, cfg_alt, i_alt):
    """Relative change of the bit-line output currents (through the loads) from i_ref to i_alt."""
    a, b = i_ref[cfg.slices[BranchKind.LOAD]], i_alt[cfg_alt.slices[BranchKind.LOAD]]
    return float(np.linalg.norm(b - a) / np.linalg.norm(a))


def entry(res, cfg, u_ref):
    """The record of one solve: status, counts and residuals, and agreement with the reference at the word and bit nodes."""
    h = res.history
    e = dict(status=res.status, iterations=None if np.isnan(res.iterations) else int(res.iterations),
             runtime=res.runtime, error=res.error)
    for key in ("kcl_residual", "kcl_residual_rel", "kvl_residual_rel", "kcl_exact_residual", "kcl_exact_residual_rel",
                "method", "spice_iterations", "halvings", "failure", "rematches", "open_cells", "contraction_bound",
                "refactor_time", "setup_time", "iteration_time", "factor_time"):
        if key in h:
            e[key] = h[key]
    if "notes" in h:
        e["notes"] = h["notes"][:6]
    if "rematch_log" in h:
        e["last_rematch"] = h["rematch_log"][-1][0] if h["rematch_log"] else None
    if res.v is not None and u_ref is not None:
        e["max_du"] = float(np.abs(cfg.line_potentials(res.v) - u_ref).max())
    return e


def feasibility_case(p, q, seed, I_s, n, with_ideal):
    """Every case at one size, seed and diode (the ideal case, which has no diode parameters, only if with_ideal):
    references first, then each case's solvers."""
    lumped = Config(p, q, seed=seed, **circuit("lumped", I_s, n))
    ref = NewtonSafeguardedSolver().solve(lumped)
    ideal_lumped = Config(p, q, seed=seed, **dict(CIRCUIT, device="1d1r"))
    ref_ideal = NewtonSolver().solve(ideal_lumped)
    assert ref.converged and ref_ideal.converged
    record = dict(p=p, q=q, seed=seed, I_s=I_s, n=n, operating_point=operating_point(lumped, ref.v),
                  output_change_vs_ideal=output_change(ideal_lumped, ref_ideal.i, lumped, ref.i), solves={})
    for case in CASES:
        if case == "ideal" and not with_ideal:
            continue
        cfg = Config(p, q, seed=seed, **circuit(case, I_s, n))
        reference = ref_ideal if case == "ideal" else ref
        u_ref = (ideal_lumped if case == "ideal" else lumped).line_potentials(reference.v)
        for solver in feasibility_solvers(case):
            record["solves"][f"{case}/{solver.name}"] = entry(run_one(solver, cfg), cfg, u_ref)
    return record


def report(rec):
    op = rec["operating_point"]
    print(f"I_s {rec['I_s']:g} n {rec['n']:g} {rec['p']}x{rec['q']} seed {rec['seed']}: reverse {op['reverse']:.2f} "
          f"sub {op['subthreshold']:.2f} knee {op['knee']:.2f} on {op['on']:.2f}; output change vs ideal "
          f"{rec['output_change_vs_ideal']:.2f}", flush=True)
    for name, e in rec["solves"].items():
        its = e["iterations"] if e["status"] == "converged" else e["status"]
        extra = [f"{k}={e[k]}" for k in ("method", "spice_iterations", "halvings", "failure", "rematches")
                 if e.get(k) not in (None, 0)]
        du = f" du={e['max_du']:.1e}" if "max_du" in e else ""
        print(f"    {name:<34} {its!s:>6}{du} " + " ".join(extra), flush=True)


def scale(args):
    """Isolated runs of every case's scaling solvers, interleaved per size and seed (see run_experiments)."""
    I_s, n = args.diode
    out_dir = args.out or ROOT / "results" / datetime.now().strftime("diode-models-scale-%Y%m%d-%H%M%S")
    (out_dir / "pickles").mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = set()
    if runs_path.exists():
        with open(runs_path) as f:
            done = {(r["case"], r["algorithm"], r["p"], r["q"], r["seed"]) for r in map(json.loads, f) if r}
    cases = args.cases
    spec = dict(started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(), sizes=args.sizes,
                seeds=args.seeds, I_s=I_s, n=n, cases=cases, timeout=args.timeout,
                circuits={case: circuit(case, I_s, n) for case in cases},
                solvers={case: [s.describe() for s in scale_solvers(case)] for case in cases}, git_commit=git_commit())
    spec_path = out_dir / "experiment.json"
    if spec_path.exists():
        spec = dict(json.loads(spec_path.read_text()), resumed=datetime.now().isoformat(timespec="seconds"))
    spec_path.write_text(json.dumps(spec, indent=2))
    with open(runs_path, "a") as log:
        for p, q in args.sizes:
            for seed in range(args.seeds):
                for case in cases:
                    for solver in scale_solvers(case):
                        if (case, solver.name, p, q, seed) in done:
                            continue
                        pickle_path = out_dir / "pickles" / f"{case}_{solver.name}_{p}x{q}_seed{seed}.pkl"
                        summary = run_isolated(solver, p, q, seed, circuit(case, I_s, n), pickle_path, args.timeout)
                        summary.update(case=case, I_s=I_s, n=n)
                        log.write(json.dumps(summary) + "\n")
                        log.flush()
                        h = summary["history"]
                        extra = " ".join(f"{k}={h[k]}" for k in ("method", "spice_iterations", "halvings", "failure",
                                                                  "rematches") if h.get(k) not in (None, 0))
                        runtime = "-" if summary["runtime"] is None else f"{summary['runtime']:9.4f}s"
                        print(f"{case:<8} {solver.name:<20} {p}x{q} seed={seed}: {summary['status']:<13} {runtime} "
                              f"iters={summary['iterations']} {extra}" + (f" ({summary['error']})" if summary["error"]
                                                                            else ""), flush=True)
    print(f"results in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("phase", choices=("feasibility", "scale"))
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=[(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)])
    ap.add_argument("--seeds", type=int, default=3, help="seeds 0..SEEDS-1 per size")
    ap.add_argument("--diodes", type=float, nargs="+", metavar="I_S N",
                    help="feasibility: (I_s, n) pairs, flattened; default " + str(DIODES))
    ap.add_argument("--diode", type=float, nargs=2, metavar=("I_S", "N"), default=(1e-6, 1.0),
                    help="scale: the diode's I_s and n")
    ap.add_argument("--cases", nargs="+", choices=SCALE_CASES, default=list(SCALE_CASES), help="scale: the cases to run")
    ap.add_argument("--timeout", type=float, default=1800, help="scale: seconds before a run is killed")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.phase == "scale":
        return scale(args)

    diodes = DIODES if args.diodes is None else list(zip(args.diodes[::2], args.diodes[1::2]))
    out_dir = args.out or ROOT / "results" / datetime.now().strftime("diode-models-feasibility-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "experiment.json").write_text(json.dumps(dict(
        started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(), sizes=args.sizes,
        seeds=args.seeds, diodes=diodes, circuit=CIRCUIT,
        solvers={case: [s.describe() for s in feasibility_solvers(case)] for case in CASES}, git_commit=git_commit()),
        indent=2))
    with open(out_dir / "records.jsonl", "a") as log:
        for I_s, n in diodes:
            for p, q in args.sizes:
                for seed in range(args.seeds):
                    t = time.perf_counter()
                    rec = feasibility_case(p, q, seed, I_s, n, (I_s, n) == diodes[0])
                    rec["wall_time"] = time.perf_counter() - t
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    report(rec)
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
