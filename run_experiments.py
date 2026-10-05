"""
Single entrypoint for comparison experiments.

For every array size and every run k, one Config is generated with seed = base_seed + k and solved by every selected
algorithm, so all algorithms see identical circuits. Each run is timed, recorded, and saved; a solver that raises or
fails to converge is recorded as such and the sweep continues. Plots of runtime, iteration count and convergence rate
vs array size are drawn at the end.

Examples:
    python run_experiments.py --sizes 8 16 32 64 --runs 5
    python run_experiments.py --sizes 16x32 64x64 --runs 10 --algorithms spingarn --tol 1e-10
    python run_experiments.py --replot results/20261005-153000

Output, in results/<timestamp>/ (or --out):
    experiment.json  sweep specification, solver settings, git commit and package versions
    runs.jsonl       one JSON record per run, appended as each run finishes
    pickles/         full Result objects (Config, branch voltages and currents), unless --no-pickle
    summary.json     per (algorithm, size) mean/std of runtime and iterations over converged runs
    runtime.png, iterations.png, convergence.png
"""
import argparse
import json
import math
import pickle
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from Config import Config
from Result import Result, CONVERGED, NOT_CONVERGED, ERROR
from algorithms import SOLVERS
from plotting import make_plots

ROOT = Path(__file__).resolve().parent


def parse_size(text):
    """'64' -> (64, 64); '32x64' -> (32, 64), as (word lines, bit lines)."""
    p, _, q = text.lower().partition("x")
    return int(p), int(q or p)


def run_one(solver, cfg):
    """Solve cfg with solver, timing only the solve call, and wrap the outcome in a Result."""
    start = time.perf_counter()
    try:
        out = solver.solve(cfg)
    except Exception as e:
        runtime = time.perf_counter() - start
        return Result(cfg, solver.name, ERROR, runtime,
                      error="".join(traceback.format_exception_only(type(e), e)).strip())
    runtime = time.perf_counter() - start
    status = CONVERGED if out.converged else NOT_CONVERGED
    iterations = out.iterations if out.converged else math.nan
    return Result(cfg, solver.name, status, runtime, iterations, out.v, out.i, out.history)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_sweep(sizes, runs, solvers, out_dir, base_seed=0, circuit=None, save_pickles=True, argv=None):
    """
    Run every solver on `runs` random circuits of each size in `sizes` and save everything to `out_dir`.

    sizes: list of (p, q). solvers: list of Solver instances. circuit: keyword overrides for Config
    (R_min, R_max, R_wire, R_source, R_load, E_range). Returns the run summaries; full Results only go to
    disk, so large sweeps do not accumulate configs and solutions in memory.
    """
    circuit = circuit or {}
    out_dir = Path(out_dir)
    (out_dir / "pickles").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "experiment.json", "w") as f:
        json.dump(dict(
            started=datetime.now().isoformat(timespec="seconds"),
            argv=argv,
            sizes=[list(s) for s in sizes],
            runs=runs,
            base_seed=base_seed,
            circuit=circuit,
            solvers=[s.describe() for s in solvers],
            git_commit=git_commit(),
            versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        ), f, indent=2)

    summaries = []
    total, done = len(sizes) * runs * len(solvers), 0
    with open(out_dir / "runs.jsonl", "a") as log:
        for p, q in sizes:
            for k in range(runs):
                cfg = Config(p, q, seed=base_seed + k, **circuit)
                for solver in solvers:
                    res = run_one(solver, cfg)
                    done += 1
                    summaries.append(res.summary())
                    log.write(json.dumps(summaries[-1]) + "\n")
                    log.flush()
                    if save_pickles:
                        with open(out_dir / "pickles" / f"{solver.name}_{p}x{q}_seed{cfg.seed}.pkl", "wb") as f:
                            pickle.dump(res, f)
                    iters = "-" if math.isnan(res.iterations) else int(res.iterations)
                    print(f"[{done}/{total}] {solver.name:<10} {p}x{q} seed={cfg.seed}: {res.status:<13} "
                          f"{res.runtime:9.4f}s  iters={iters}" + (f"  ({res.error})" if res.error else ""))
    return summaries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=[(8, 8), (16, 16), (32, 32)],
                    help="array sizes, N for NxN or PxQ (word lines x bit lines)")
    ap.add_argument("--runs", type=int, default=5, help="random circuits per size; metrics are averaged over them")
    ap.add_argument("--algorithms", nargs="+", choices=list(SOLVERS), default=list(SOLVERS))
    ap.add_argument("--seed", type=int, default=0, help="run k uses seed SEED + k")
    ap.add_argument("--tol", type=float, default=1e-9, help="convergence tolerance passed to every solver")
    ap.add_argument("--max-iter", type=int, default=10_000, help="iteration budget passed to every solver")
    ap.add_argument("--circuit", type=json.loads, default={},
                    help='JSON overrides for Config, e.g. \'{"R_wire": 10, "E_range": [0.1, 0.3]}\'')
    ap.add_argument("--out", type=Path, help="output directory (default results/<timestamp>)")
    ap.add_argument("--no-pickle", action="store_true", help="skip full Result pickles (JSON records still saved)")
    ap.add_argument("--replot", type=Path, metavar="DIR", help="redraw plots from an existing results directory")
    args = ap.parse_args(argv)

    if args.replot:
        make_plots(args.replot)
        print(f"plots redrawn in {args.replot}")
        return

    out_dir = args.out or ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    solvers = [SOLVERS[name](tol=args.tol, max_iterations=args.max_iter) for name in args.algorithms]
    run_sweep(args.sizes, args.runs, solvers, out_dir, args.seed, args.circuit, not args.no_pickle,
              argv=sys.argv if argv is None else argv)
    make_plots(out_dir, [s.name for s in solvers])
    print(f"results and plots saved to {out_dir}")


if __name__ == "__main__":
    main()
