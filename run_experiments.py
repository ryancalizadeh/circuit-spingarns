"""
Single entrypoint for comparison experiments.

For every array size and every run k, one Config is generated with seed = base_seed + k and solved by every selected
algorithm, so all algorithms see identical circuits. Each run executes in its own process, so a solver that exhausts
memory or crashes natively is recorded as "crashed" (and one that exceeds --timeout as "timeout") while the sweep
carries on. A solver that raises or fails to converge is recorded as such too. Plots of runtime, iteration count and
convergence rate vs array size are drawn at the end.

Every finished run is written to runs.jsonl immediately, and summary.json and the plots are refreshed after each array
size, so an interrupted sweep keeps its data; --resume continues it, skipping runs that already have a record.

Examples:
    python run_experiments.py --sizes 8 16 32 64 --runs 5
    python run_experiments.py --sizes 16x32 64x64 --runs 10 --algorithms spingarn --tol 1e-10
    python run_experiments.py --sizes 64 128 --algorithms spice --options '{"spice": {"matrix_solver": "sparse"}}'
    python run_experiments.py --sizes 256 512 1024 --runs 3 --timeout 3600
    python run_experiments.py --resume results/20261005-153000
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
import multiprocessing
import os
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
from Result import Result, CONVERGED, NOT_CONVERGED, ERROR, CRASHED, TIMEOUT
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


def warm_up(solver, circuit):
    """One untimed solve of a tiny circuit, so one-time costs (library loading, first-call setup) are not charged to
    the timed run. A solver that fails here fails again in the timed run, where it is recorded."""
    try:
        solver.solve(Config(2, 2, seed=0, **circuit))
    except Exception:
        pass


def save_pickle(res, path):
    """Pickle the full Result via a temporary file, so a pickle on disk is always complete."""
    tmp = Path(f"{path}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    tmp.replace(path)


def solve_and_save(solver, p, q, seed, circuit, pickle_path, report=None):
    """
    Generate the circuit, solve it, and return the run summary. The summary goes to `report` (if given) before the
    full Result is pickled to `pickle_path` (if set), so a failure while pickling cannot lose the measurement.
    """
    res = run_one(solver, Config(p, q, seed=seed, **circuit))
    summary = res.summary()
    if report:
        report(summary)
    if pickle_path:
        save_pickle(res, pickle_path)
    return summary


def _isolated_run(conn, solver, p, q, seed, circuit, pickle_path):
    warm_up(solver, circuit)
    solve_and_save(solver, p, q, seed, circuit, pickle_path, report=conn.send)
    conn.close()


def failure_summary(solver, p, q, seed, circuit, status, error, wall_time):
    """Run record for a solver process that never returned a Result, in the same format as Result.summary()."""
    params = Config(1, 1, seed=seed, **circuit).params()  # full parameter set without building the p x q array
    params.update(num_word_lines=p, num_bit_lines=q)
    return dict(algorithm=solver.name, p=p, q=q, seed=seed, status=status, runtime=None, iterations=None,
                error=error, history=dict(wall_time=wall_time), config=params)


def run_isolated(solver, p, q, seed, circuit, pickle_path, timeout=None):
    """
    solve_and_save in a fresh process. If the process dies before reporting (out of memory, a native crash) the run is
    recorded as CRASHED; if it is still running after `timeout` seconds (process start-up included) it is killed and
    recorded as TIMEOUT. Only the small summary crosses back; the circuit is generated and pickled in the child.
    """
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_isolated_run, args=(sender, solver, p, q, seed, circuit, pickle_path), daemon=True)
    start = time.perf_counter()
    proc.start()
    sender.close()  # the child holds the only write end now, so its death shows up as EOF here
    try:
        if receiver.poll(timeout):
            summary = receiver.recv()
            proc.join()  # the child pickles the full Result after reporting; wait for it
            return summary
        status, error = TIMEOUT, f"no result after {timeout} s"
    except EOFError:
        proc.join()
        status, error = CRASHED, f"solver process exited with code {proc.exitcode} ({proc.exitcode & 0xFFFFFFFF:#x})"
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        receiver.close()
    return failure_summary(solver, p, q, seed, circuit, status, error, time.perf_counter() - start)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_records(out_dir):
    path = Path(out_dir) / "runs.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_sweep(sizes, runs, solvers, out_dir, base_seed=0, circuit=None, save_pickles=True, args=None,
              isolate=True, timeout=None, resume=False):
    """
    Run every solver on `runs` random circuits of each size in `sizes` and save everything to `out_dir`.

    sizes: list of (p, q). solvers: list of Solver instances. circuit: keyword overrides for Config
    (R_min, R_max, R_wire, R_source, R_load, E_range). isolate: run each solve in its own process (see run_isolated);
    timeout only applies then. resume: keep out_dir's records and skip the runs they cover. Returns the run summaries
    of this call; full Results only go to disk, so large sweeps do not accumulate configs and solutions in memory.
    """
    circuit = circuit or {}
    out_dir = Path(out_dir)
    (out_dir / "pickles").mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "experiment.json"
    if resume and spec_path.exists():
        spec = json.loads(spec_path.read_text())
        spec.setdefault("resumed", []).append(datetime.now().isoformat(timespec="seconds"))
    else:
        spec = dict(
            started=datetime.now().isoformat(timespec="seconds"),
            args=args,
            sizes=[list(s) for s in sizes],
            runs=runs,
            base_seed=base_seed,
            circuit=circuit,
            solvers=[s.describe() for s in solvers],
            isolated=isolate,
            timeout=timeout,
            git_commit=git_commit(),
            versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        )
    spec_path.write_text(json.dumps(spec, indent=2))

    if not resume and load_records(out_dir):
        raise FileExistsError(f"{out_dir} already holds run records; resume it or choose another output directory")
    done_before = {(r["algorithm"], r["p"], r["q"], r["seed"]) for r in load_records(out_dir)}
    if not isolate:
        for solver in solvers:
            warm_up(solver, circuit)

    summaries = []
    total, done = len(sizes) * runs * len(solvers), 0
    with open(out_dir / "runs.jsonl", "a") as log:
        for p, q in sizes:
            for k in range(runs):
                seed = base_seed + k
                for solver in solvers:
                    done += 1
                    if (solver.name, p, q, seed) in done_before:
                        continue
                    pickle_path = out_dir / "pickles" / f"{solver.name}_{p}x{q}_seed{seed}.pkl" if save_pickles else None
                    if isolate:
                        summary = run_isolated(solver, p, q, seed, circuit, pickle_path, timeout)
                    else:
                        summary = solve_and_save(solver, p, q, seed, circuit, pickle_path)
                    summaries.append(summary)
                    log.write(json.dumps(summary) + "\n")
                    log.flush()
                    os.fsync(log.fileno())  # the record survives even if the machine goes down next
                    runtime = "-" if summary["runtime"] is None else f"{summary['runtime']:9.4f}s"
                    iters = "-" if summary["iterations"] is None else summary["iterations"]
                    print(f"[{done}/{total}] {solver.name:<10} {p}x{q} seed={seed}: {summary['status']:<13} "
                          f"{runtime}  iters={iters}" + (f"  ({summary['error']})" if summary["error"] else ""),
                          flush=True)
            checkpoint(out_dir, solvers)
    return summaries


def checkpoint(out_dir, solvers):
    """Refresh summary.json and the plots from every record so far; a plotting failure never stops the sweep."""
    try:
        make_plots(out_dir, [s.name for s in solvers])
    except Exception as e:
        print(f"warning: could not refresh summary/plots: {e}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=[(8, 8), (16, 16), (32, 32)],
                    help="array sizes, N for NxN or PxQ (word lines x bit lines)")
    ap.add_argument("--runs", type=int, default=5, help="random circuits per size; metrics are averaged over them")
    ap.add_argument("--algorithms", nargs="+", choices=list(SOLVERS), default=list(SOLVERS))
    ap.add_argument("--seed", type=int, default=0, help="run k uses seed SEED + k")
    ap.add_argument("--tol", type=float,
                    help="convergence tolerance for every solver, solver-specific meaning (SPICE: RELTOL); "
                         "default: each solver's own")
    ap.add_argument("--max-iter", type=int,
                    help="iteration budget for every solver (SPICE: ITL1); default: each solver's own")
    ap.add_argument("--options", type=json.loads, default={},
                    help='per-algorithm solver options as JSON, e.g. \'{"spice": {"matrix_solver": "sparse"}}\'')
    ap.add_argument("--circuit", type=json.loads, default={},
                    help='JSON overrides for Config, e.g. \'{"R_wire": 10, "E_range": [0.1, 0.3]}\'')
    ap.add_argument("--timeout", type=float, help="seconds before a run is killed and recorded as a timeout")
    ap.add_argument("--no-isolation", action="store_true",
                    help="run solves in this process: faster for small sweeps, but a crash ends the sweep")
    ap.add_argument("--out", type=Path, help="output directory (default results/<timestamp>)")
    ap.add_argument("--no-pickle", action="store_true", help="skip full Result pickles (JSON records still saved)")
    ap.add_argument("--resume", type=Path, metavar="DIR",
                    help="continue an interrupted sweep in DIR with its original arguments")
    ap.add_argument("--replot", type=Path, metavar="DIR", help="redraw plots from an existing results directory")
    return ap


def main(argv=None):
    ap = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    args = ap.parse_args(argv)

    if args.replot:
        make_plots(args.replot)
        print(f"plots redrawn in {args.replot}")
        return

    out_dir = args.out or ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    resume = args.resume is not None
    if resume:
        out_dir = args.resume
        argv = json.loads((out_dir / "experiment.json").read_text()).get("args")
        if argv is None:
            ap.error(f"{out_dir}/experiment.json records no command-line arguments to resume with")
        args = ap.parse_args(argv)

    unknown = set(args.options) - set(args.algorithms)
    if unknown:
        ap.error(f"--options given for algorithms not being run: {', '.join(sorted(unknown))}")
    solvers = [SOLVERS[name](tol=args.tol, max_iterations=args.max_iter, **args.options.get(name, {}))
               for name in args.algorithms]
    run_sweep(args.sizes, args.runs, solvers, out_dir, args.seed, args.circuit, not args.no_pickle, args=argv,
              isolate=not args.no_isolation, timeout=args.timeout, resume=resume)
    print(f"results and plots saved to {out_dir}")


if __name__ == "__main__":
    main()
