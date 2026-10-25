"""
Network models sweep: SPICE against Spingarn with fixed port resistances, matched at zero bias or over the sector, on
three models of the same crossbar's cells:

    memristor      the tanh memristor alone (device "memristor");
    1d1r_lumped    an ideal diode in series with it, one branch per cell: i = g_m(max(v, 0)) (device "1d1r");
    1d1r_separate  the same circuit with each diode on a branch of its own, through an internal node (device "1d1r",
                   internal_node=True): the diode is exact (set-valued) in Spingarn and uramp(V) / 1 mohm in SPICE;
    1d1r_shockley_separate
                   as 1d1r_separate with a Shockley diode, I_s = 1e-9 A, n = 1 (the user's choice, 2026-10-19), a D
                   element in SPICE (not the exact Shockley law: memory/diode-models-findings.md).

All four use report_nonsmooth.md's 1D1R crossbar (E in [-1, 1.5] V, per-cell R_min 1-10 kohm and R_max 50-200 kohm,
wires 5 ohm, sources 50 ohm, loads 1 kohm), so only the cell model changes between cases.

Solvers (the user's choice, 2026-10-19):
    spice               ngspice-47 through PySpice, KLU, RELTOL 1e-3, ITL1 100 (SpiceSolver's defaults);
    spice_exp           the same with Shockley diodes as behavioural exp() sources (SpiceExpSolver), the exact law
                        without the D element's junction limiting (the user's choice for the Shockley case, 2026-10-19;
                        not in the default --algorithms);
    spingarn_zero_bias  cells matched at R_min, the memristor's zero-bias incremental resistance (gamma "r_lo");
    spingarn_sector     cells matched over the memristor's sector, sqrt(R_min R_max) (gamma "geometric").
Both Spingarn variants keep Gamma fixed (no adaptive re-matching) and factorize with CHOLMOD in its own ordering (AMD in
cvxopt's build) rather than the geometric nested dissection. alpha is 1 on memristors and 0.8 on 1D1R cells, where
alpha = 1 never converges (memory/diode-1d1r-findings.md). Separate diodes get the R_min port in both variants: an
ideal diode's sector [0, inf) has no finite end to match, so "sector" matching applies to the memristors only.

Every run executes in its own process (run_experiments.run_isolated), sizes ascending, the cases' solvers interleaved per
size and seed. Each record in runs.jsonl is run_isolated's summary plus case, machine and costs (see run_costs): the
one-time cost, the total and per-iteration cost of the iterations, and for SPICE its fallbacks. The solvers' histories
hold the full profile (SpingarnSolver and SpiceSolver docstrings); every run also prints a one-line profile, which goes
to sweep.log too.

Skipping: once a solver fails a case at some size (crash, timeout, Spingarn at its iteration cap, or SPICE with the
signature of KLU's size limit: every fallback failed and no numeric factorization ran), its remaining seeds at that size
and all larger sizes of that case are skipped and listed in skipped.jsonl. --spice-max-size skips SPICE above a size.
--runtime-limit-at skips, at a size, every solver and case whose runtime extrapolated from the two largest smaller sizes
exceeds a limit (the user's rule for 1024x1024, 2026-10-19: skip what would take over an hour).

--runtime-cap kills a run that is still solving that many seconds into its timed solve and records it as a timeout,
"failed to converge within the time limit" (the user's rule of 2026-10-24: 1600 s); a run that converged only after the
cap is recorded the same way (status timeout, history["converged_after"]). A timeout ends the solver's sweep of the case
at larger sizes only: its other seeds at that size still run, since they may finish under the cap, unless the solver is
named in --skip-seeds-after-timeout (all solvers if it is given without names).

    python network_models_experiment.py sweep --sizes 16 32 64 128 256 512 1024 --seeds 3 --spice-max-size 512 \
        --runtime-limit-at 1024=3600
    python network_models_experiment.py sweep --sizes 16 32 64 128 256 512 1024 --seeds 10 --spice-max-size 512 \
        --runtime-cap 1600 --cases memristor --algorithms spice spingarn_zero_bias
    python network_models_experiment.py sweep --out results/network-models --resume
    python network_models_experiment.py summarize results/network-models
    python network_models_experiment.py export results/network-models    # runtime.csv, runs.csv (e.g. for pgfplots)

Output, in results/network-models-<timestamp>/ (or --out): experiment.json, runs.jsonl, skipped.jsonl, sweep.log,
pickles/ (full Results), and from summarize (also run after each size) summary.json and summary.md: per case, solver and
size, medians and ranges over converged runs, and agreement between the solvers' solutions at the word and bit nodes.
"""
import argparse
import json
import math
import os
import pickle
import platform
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from Result import CONVERGED, CRASHED, NOT_CONVERGED, TIMEOUT
from algorithms.spice import SpiceExpSolver, SpiceSolver
from algorithms.spingarns import DIODE_GAMMA_RULES, SpingarnSolver
from run_experiments import git_commit, parse_size, run_isolated

ROOT = Path(__file__).resolve().parent
CIRCUIT = dict(E_range=(-1.0, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
CASES = {
    "memristor": dict(CIRCUIT),
    "1d1r_lumped": dict(CIRCUIT, device="1d1r"),
    "1d1r_separate": dict(CIRCUIT, device="1d1r", internal_node=True),
    "1d1r_shockley_separate": dict(CIRCUIT, device="1d1r_shockley", internal_node=True, I_s=1e-9, n=1.0),
}
ALPHA = {"memristor": 1.0, "1d1r_lumped": 0.8, "1d1r_separate": 0.8, "1d1r_shockley_separate": 0.8}
SOLVER_NAMES = ("spice", "spice_exp", "spingarn_zero_bias", "spingarn_sector")
SPICE_NAMES = ("spice", "spice_exp")
DEFAULT_ALGORITHMS = ("spice", "spingarn_zero_bias", "spingarn_sector")  # also what specs without "algorithms" ran
# Spingarn's iteration cap: above the ~7,800 iterations projected for fixed matching on 1D1R at 1024x1024 (x2.8 per
# doubling from 941 at 256, 2026-10-19), and meant to bind before --timeout so a run that hits it keeps its history.
MAX_ITERATIONS = 20_000


def named(solver, name):
    """solver under another name, so several settings of one class can share a sweep."""
    solver.name = name
    return solver


def diode_suffix(diode_gamma):
    """The name suffix of Spingarn solvers whose separate diodes get port rule diode_gamma: none for the default
    "r_min", so the names of the first sweep stand; e.g. "_dgeometric" or "_d10000" otherwise."""
    if diode_gamma == "r_min":
        return ""
    return f"_d{diode_gamma}" if isinstance(diode_gamma, str) else f"_d{diode_gamma:g}"


def solvers(case, max_iterations=MAX_ITERATIONS, diode_gamma="r_min", algorithms=DEFAULT_ALGORITHMS):
    """The sweep's solvers for a case, those of algorithms (names in SOLVER_NAMES) only. diode_gamma is the port rule
    of separate diodes (diode_port_resistances); lumped cells have no diode branch, so it changes nothing there."""
    spingarn = dict(alpha=ALPHA[case], ordering="builtin", max_iterations=max_iterations, diode_gamma=diode_gamma)
    suffix = diode_suffix(diode_gamma)
    make = dict(spice=SpiceSolver, spice_exp=SpiceExpSolver,
                spingarn_zero_bias=lambda: SpingarnSolver(gamma="r_lo", **spingarn),
                spingarn_sector=lambda: SpingarnSolver(gamma="geometric", **spingarn))
    return [make[name]() if name in SPICE_NAMES else named(make[name](), name + suffix)
            for name in SOLVER_NAMES if name in algorithms]


def base_name(algorithm):
    """The SOLVER_NAMES entry that algorithm is, or extends by a diode suffix (diode_suffix); the longest such, so that
    spice_exp is a solver of its own rather than spice with a suffix. None for other names."""
    bases = [name for name in SOLVER_NAMES if algorithm == name or algorithm.startswith(name + "_")]
    return max(bases, key=len) if bases else None


def is_spice(algorithm):
    return base_name(algorithm) in SPICE_NAMES


def solver_order(algorithm):
    """Sort key: SOLVER_NAMES order by base name, suffixed variants (diode_suffix) after their base."""
    base = base_name(algorithm)
    return (len(SOLVER_NAMES) if base is None else SOLVER_NAMES.index(base)), algorithm


def short_name(algorithm):
    """The CSV column stem of a solver: CSV_SOLVERS's short name with any diode suffix, e.g. "zb_dgeometric"."""
    base = base_name(algorithm)
    return algorithm if base is None else CSV_SOLVERS.get(base, base) + algorithm[len(base):]


def run_costs(summary):
    """
    Where a run's time went, in seconds (None where not recorded):

        iterations_run   iterations performed, converged or not (SPICE: ngspice's total, fallback iterations included);
        one_time         Spingarn: setup (assembly, ordering, factorization); SPICE: netlist text, ngspice's netlist
                         load, and KLU's reorder (its symbolic analysis);
        iterations_total Spingarn: the iteration loop; SPICE: ngspice's analysis time less the reorder;
        per_iteration    iterations_total / iterations_run;
        post             SPICE only: reading the operating point back and the KCL check;
        unaccounted      the rest of the runtime (Python glue, ngspice's bookkeeping).
    """
    h = summary.get("history") or {}
    runtime = summary.get("runtime")
    if is_spice(summary["algorithm"]):
        its = h.get("spice_iterations")
        reorder = h.get("matrix_reorder_time")
        one_time = (sum(h[k] for k in ("netlist_time", "load_time") if k in h) + (reorder or 0.0)
                    if "load_time" in h else None)
        total = h["spice_analysis_time"] - (reorder or 0.0) if "spice_analysis_time" in h else None
        post = sum(h.get(k, 0.0) for k in ("extract_time", "verify_time"))
    else:
        its = len(h["kcl_history"]) if "kcl_history" in h else None
        one_time, total, post = h.get("setup_time"), h.get("iteration_time"), 0.0
    per_iteration = total / its if its and total is not None else None
    parts = [one_time, total]
    unaccounted = runtime - sum(parts) - post if runtime is not None and None not in parts else None
    return dict(iterations_run=its, one_time=one_time, iterations_total=total, per_iteration=per_iteration, post=post,
                unaccounted=unaccounted)


def klu_size_limit(summary):
    """Whether a SPICE run failed the way ngspice's KLU fails beyond its size limit (memory/ngspice-baseline-findings):
    every fallback failed and no numeric factorization ever ran."""
    h = summary.get("history") or {}
    return h.get("method") == "failed" and h.get("matrix_factor_time") == 0.0


def failure_reason(summary):
    """Why this run ends its solver's sweep of the case (see the module docstring), or None if it doesn't."""
    status = summary["status"]
    if status == TIMEOUT and (summary.get("history") or {}).get("runtime_cap") is not None:
        return f"over the {summary['history']['runtime_cap']:g} s runtime cap"
    if status in (CRASHED, TIMEOUT):
        return status
    if is_spice(summary["algorithm"]):
        return "KLU size limit" if klu_size_limit(summary) else None
    if status == NOT_CONVERGED:
        return "iteration cap"
    return None


def profile_line(rec):
    """One line on where a run's time and memory went."""
    h, c = rec["history"], rec["costs"]
    ms = lambda x: "-" if x is None else f"{1e3 * x:.1f}"
    s = lambda x: "-" if x is None else f"{x:.3g}"
    mem = "-" if rec.get("peak_memory") is None else f"{rec['peak_memory'] / 1e9:.2f} GB"
    cpu = (f" cpu/wall {h['cpu_time'] / rec['runtime']:.2f}" if h.get("cpu_time") is not None and rec["runtime"]
           else "")
    if is_spice(rec["algorithm"]):
        its = c["iterations_run"]
        per = lambda k: ms(h[k] / its) if its and k in h else "-"
        fallbacks = ", ".join(f"{stage} {outcome}" for stage, outcome in h.get("fallbacks", [])) or "none"
        return (f"    one-time {s(c['one_time'])} s (netlist {s(h.get('netlist_time'))}, load {s(h.get('load_time'))}, "
                f"reorder {s(h.get('matrix_reorder_time'))}) | per Newton it {ms(c['per_iteration'])} ms (load "
                f"{per('matrix_load_time')}, factor {per('matrix_factor_time')}, solve {per('matrix_solve_time')}) | "
                f"method {h.get('method')}, fallbacks: {fallbacks} | eqs {h.get('equations'):.0f}, fill "
                f"{h.get('fill_in', 0):.3g} | {mem}{cpu}" if "equations" in h else f"    {mem}{cpu}")
    its = c["iterations_run"]
    per = lambda k: ms(h[k] / its) if its and k in h else "-"
    return (f"    one-time {s(c['one_time'])} s (assembly {s(h.get('assembly_time'))}, symbolic "
            f"{s(h.get('symbolic_time'))}, numeric {s(h.get('numeric_time'))}; lnz {h.get('lnz', 0):.3g}) | "
            f"per it {ms(c['per_iteration'])} ms (resolvent {per('resolvent_time')}, network {per('network_time')} "
            f"[solve {per('triangular_solve_time')}], check {per('check_time')}, update {per('update_time')}) | "
            f"newton steps/it {h.get('newton_steps', 0) / its if its else 0:.2f} | {mem}{cpu}")


def parse_seeds_at(text):
    size, _, seeds = text.partition("=")
    return parse_size(size), int(seeds)


def parse_limit_at(text):
    size, _, seconds = text.partition("=")
    return parse_size(size), float(seconds)


def projected_runtime(records, case, algorithm, p, q):
    """
    The runtime of (case, algorithm) at p x q extrapolated from its two largest smaller sizes with converged runs, as a
    power law in the number of cells through their median runtimes; with the sizes it came from, or None if there are
    fewer than two such sizes.
    """
    by_size = defaultdict(list)
    for r in records:
        if (r["case"], r["algorithm"]) == (case, algorithm) and r["status"] == CONVERGED and r["p"] * r["q"] < p * q:
            by_size[r["p"] * r["q"]].append(r["runtime"])
    if len(by_size) < 2:
        return None
    (n1, t1), (n2, t2) = [(n, statistics.median(ts)) for n, ts in sorted(by_size.items())[-2:]]
    return t2 * (p * q / n2) ** (math.log(t2 / t1) / math.log(n2 / n1)), (n1, n2)


def sweep(args):
    out_dir = args.out or ROOT / "results" / datetime.now().strftime("network-models-%Y%m%d-%H%M%S")
    (out_dir / "pickles").mkdir(parents=True, exist_ok=True)
    runs_path, skipped_path, spec_path = out_dir / "runs.jsonl", out_dir / "skipped.jsonl", out_dir / "experiment.json"
    records = [json.loads(line) for line in open(runs_path) if line.strip()] if runs_path.exists() else []
    if records and not args.resume:
        raise SystemExit(f"{out_dir} already holds run records; pass --resume to continue it")
    if args.resume and spec_path.exists():
        spec = json.loads(spec_path.read_text())
        spec.setdefault("resumed", []).append(datetime.now().isoformat(timespec="seconds"))
        sizes = [tuple(s) for s in spec["sizes"]]
        seeds_at = {tuple(json.loads(k)): v for k, v in spec["seeds_at"].items()}
        limits = {tuple(json.loads(k)): v for k, v in spec["runtime_limit_at"].items()}
        cases, seeds, timeout = spec["cases"], spec["seeds"], spec["timeout"]
        spice_max, skip, max_iterations = spec["spice_max_size"], spec["skip_after_failure"], spec["max_iterations"]
        runtime_cap = spec.get("runtime_cap")
        skip_seeds_after_timeout = spec.get("skip_seeds_after_timeout", True)  # older sweeps skipped them
        diode_gamma, algorithms = spec.get("diode_gamma", "r_min"), spec.get("algorithms", list(DEFAULT_ALGORITHMS))
        reference_dir = spec.get("reference_dir")
    else:
        sizes, seeds_at, limits, cases = args.sizes, dict(args.seeds_at), dict(args.runtime_limit_at), args.cases
        seeds, timeout, spice_max, skip = args.seeds, args.timeout, args.spice_max_size, not args.no_skip
        max_iterations, diode_gamma, algorithms = args.max_iterations, args.diode_gamma, args.algorithms
        runtime_cap = args.runtime_cap
        skip_seeds_after_timeout = (False if args.skip_seeds_after_timeout is None else
                                    args.skip_seeds_after_timeout or True)
        reference_dir = None if args.reference_dir is None else str(args.reference_dir)
        spec = dict(started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(),
                    processor=platform.processor(), cpu_count=os.cpu_count(), sizes=[list(s) for s in sizes],
                    seeds=seeds, seeds_at={json.dumps(list(k)): v for k, v in seeds_at.items()},
                    runtime_limit_at={json.dumps(list(k)): v for k, v in limits.items()}, cases=cases,
                    timeout=timeout, runtime_cap=runtime_cap, skip_seeds_after_timeout=skip_seeds_after_timeout,
                    spice_max_size=spice_max, skip_after_failure=skip, max_iterations=max_iterations,
                    diode_gamma=diode_gamma, algorithms=algorithms, reference_dir=reference_dir,
                    circuits={case: CASES[case] for case in cases},
                    solvers={case: [s.describe() for s in solvers(case, max_iterations, diode_gamma, algorithms)]
                             for case in cases},
                    git_commit=git_commit())
    spec_path.write_text(json.dumps(spec, indent=2))

    def ends_size(r):
        """Whether r's failure also skips its solver's other seeds at r's size (see the module docstring)."""
        policy = skip_seeds_after_timeout
        return r["status"] != TIMEOUT or policy is True or (isinstance(policy, list) and r["algorithm"] in policy)

    done = {(r["case"], r["algorithm"], r["p"], r["q"], r["seed"]) for r in records}
    skipped_before = ({tuple(s[k] for k in ("case", "algorithm", "p", "q", "seed"))
                       for s in map(json.loads, open(skipped_path)) if s} if skipped_path.exists() else set())
    dead = {}  # (case, solver) -> (cells, reason, same size): the solver's sweep of that case ended at that size
    for r in records:
        reason = failure_reason(r)
        if skip and reason and (r["case"], r["algorithm"]) not in dead:
            dead[(r["case"], r["algorithm"])] = (r["p"] * r["q"], reason, ends_size(r))

    def log(line):
        print(line, flush=True)
        with open(out_dir / "sweep.log", "a") as f:
            f.write(line + "\n")

    log(f"== {datetime.now().isoformat(timespec='seconds')} on {platform.node()}: sizes {sizes}, seeds {seeds} "
        f"(at {seeds_at}), cases {cases}, timeout {timeout} s, runtime cap {runtime_cap} s, SPICE up to {spice_max}")
    for p, q in sizes:
        for seed in range(seeds_at.get((p, q), seeds)):
            for case in cases:
                for solver in solvers(case, max_iterations, diode_gamma, algorithms):
                    key = (case, solver.name, p, q, seed)
                    if key in done:
                        continue
                    reason = None
                    if (case, solver.name) in dead:
                        cells, why, same_size = dead[(case, solver.name)]
                        reason = (f"{why} at {int(cells ** 0.5)}x{int(cells ** 0.5)}"
                                  if cells < p * q or (same_size and cells == p * q) else None)
                    if is_spice(solver.name) and spice_max is not None and p * q > spice_max[0] * spice_max[1]:
                        reason = f"above --spice-max-size {spice_max[0]}x{spice_max[1]}"
                    if not reason and (p, q) in limits:
                        projection = projected_runtime(records, case, solver.name, p, q)
                        if projection and projection[0] > limits[(p, q)]:
                            (n1, n2), side = projection[1], lambda n: f"{math.isqrt(n)}x{math.isqrt(n)}"
                            reason = (f"projected {projection[0] / 60:.0f} min from {side(n1)} and {side(n2)}, over "
                                      f"the {limits[(p, q)] / 60:.0f} min limit")
                    if reason:
                        if key not in skipped_before:
                            with open(skipped_path, "a") as f:
                                f.write(json.dumps(dict(case=case, algorithm=solver.name, p=p, q=q, seed=seed,
                                                        reason=reason)) + "\n")
                            skipped_before.add(key)
                            log(f"{case:<13} {solver.name:<29} {p}x{q} seed={seed}: skipped ({reason})")
                        continue
                    pickle_path = out_dir / "pickles" / f"{case}_{solver.name}_{p}x{q}_seed{seed}.pkl"
                    rec = run_isolated(solver, p, q, seed, CASES[case], pickle_path, timeout, runtime_cap)
                    if runtime_cap is not None and rec["runtime"] is not None and rec["runtime"] > runtime_cap:
                        rec["history"].update(converged_after=rec["runtime"] if rec["status"] == CONVERGED else None,
                                              runtime_cap=runtime_cap)
                        rec.update(status=TIMEOUT, error=f"runtime {rec['runtime']:.1f} s, over the runtime cap "
                                                         f"{runtime_cap:g} s ({rec['status']})")
                    rec.update(case=case, machine=platform.node(), costs=run_costs(rec))
                    records.append(rec)
                    with open(runs_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    reason = failure_reason(rec)
                    if skip and reason and (case, solver.name) not in dead:
                        dead[(case, solver.name)] = (p * q, reason, ends_size(rec))
                    runtime = "-" if rec["runtime"] is None else f"{rec['runtime']:9.3f}s"
                    its = rec["iterations"] if rec["iterations"] is not None else rec["costs"]["iterations_run"]
                    log(f"{case:<13} {solver.name:<29} {p}x{q} seed={seed}: {rec['status']:<13} {runtime} its={its}"
                        + (f" ({rec['error']})" if rec["error"] else "") + (f" [ends sweep: {reason}]" if reason
                                                                             else ""))
                    log(profile_line(rec))
        try:
            summarize(out_dir, reference_dir)
        except Exception as e:  # a summary failure never stops the sweep
            log(f"warning: could not summarize: {e!r}")
    log(f"results in {out_dir}")


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def agreement(out_dir, records, reference_dir=None):
    """Max |difference| in word and bit node potentials (V) between each converged solution and the reference: SPICE's
    (spice_exp's, the exact Shockley law, before the D element's spice) if it converged, else spingarn_zero_bias's, else (with reference_dir, another sweep of the same circuits) SPICE's
    solution in reference_dir. Read from the pickles; {} without them."""
    out = {}
    groups = defaultdict(dict)
    for r in records:
        if r["status"] == CONVERGED:
            groups[(r["case"], r["p"], r["q"], r["seed"])][r["algorithm"]] = r
    for (case, p, q, seed), by_alg in groups.items():
        paths = {a: out_dir / "pickles" / f"{case}_{a}_{p}x{q}_seed{seed}.pkl" for a in by_alg}
        ref_name = next((a for a in ("spice_exp", "spice", "spingarn_zero_bias") if a in by_alg), None)
        ref_path = paths.get(ref_name)
        if ref_name is None and reference_dir is not None:
            ref_name = f"spice ({Path(reference_dir).name})"
            ref_path = Path(reference_dir) / "pickles" / f"{case}_spice_{p}x{q}_seed{seed}.pkl"
        if (ref_path is None or not ref_path.exists() or len(paths) < (2 if ref_name in paths else 1)
                or not all(path.exists() for path in paths.values())):
            continue
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)
        u_ref = ref.config.line_potentials(ref.v)
        for alg, path in paths.items():
            if alg == ref_name:
                continue
            with open(path, "rb") as f:
                res = pickle.load(f)
            du = float(np.abs(res.config.line_potentials(res.v) - u_ref).max())
            out.setdefault(f"{case}/{alg}/{p}x{q}", {})[str(seed)] = dict(reference=ref_name, max_du=du)
    return out


def summarize(out_dir, reference_dir=None):
    """summary.json and summary.md from runs.jsonl: per case, solver and size, statuses, and medians and ranges over
    converged runs of runtime, iterations, one-time and per-iteration cost, peak memory and the profile, and the
    solutions' agreement (see agreement; reference_dir defaults to the sweep's own, from experiment.json)."""
    out_dir = Path(out_dir)
    spec = json.loads((out_dir / "experiment.json").read_text())
    reference_dir = reference_dir or spec.get("reference_dir")
    records = [json.loads(line) for line in open(out_dir / "runs.jsonl") if line.strip()]
    skipped = ([json.loads(line) for line in open(out_dir / "skipped.jsonl") if line.strip()]
               if (out_dir / "skipped.jsonl").exists() else [])
    groups = defaultdict(list)
    for r in records:
        groups[(r["case"], r["algorithm"], r["p"], r["q"])].append(r)
    rows = []
    for (case, alg, p, q), rs in sorted(groups.items(), key=lambda kv: (list(CASES).index(kv[0][0]),
                                                                         solver_order(kv[0][1]),
                                                                         kv[0][2] * kv[0][3])):
        ok = [r for r in rs if r["status"] == CONVERGED]
        hist = lambda key: [r["history"].get(key) for r in ok]
        per_it = lambda key: [r["history"][key] / r["costs"]["iterations_run"] for r in ok
                              if key in r["history"] and r["costs"]["iterations_run"]]
        row = dict(case=case, algorithm=alg, p=p, q=q, runs=len(rs), converged=len(ok),
                   statuses=sorted({r["status"] for r in rs}),
                   skipped=sum(1 for s in skipped if (s["case"], s["algorithm"], s["p"], s["q"]) == (case, alg, p, q)),
                   runtime=_stats([r["runtime"] for r in ok]), iterations=_stats([r["iterations"] for r in ok]),
                   one_time=_stats([r["costs"]["one_time"] for r in ok]),
                   iterations_total=_stats([r["costs"]["iterations_total"] for r in ok]),
                   per_iteration=_stats([r["costs"]["per_iteration"] for r in ok]),
                   peak_memory_gb=_stats([r["peak_memory"] / 1e9 for r in ok if r.get("peak_memory") is not None]),
                   cpu_over_wall=_stats([r["history"]["cpu_time"] / r["runtime"] for r in ok
                                         if r["history"].get("cpu_time") is not None and r["runtime"]]),
                   kcl_residual_rel=_stats(hist("kcl_residual_rel")), kcl_residual=_stats(hist("kcl_residual")),
                   failures=[dict(seed=r["seed"], status=r["status"], error=r["error"],
                                  method=r["history"].get("method"), wall_time=r["history"].get("wall_time"))
                             for r in rs if r["status"] != CONVERGED])
        if is_spice(alg):
            row.update(methods=sorted({str(r["history"].get("method")) for r in rs}),
                       fallbacks=[dict(seed=r["seed"], fallbacks=r["history"].get("fallbacks"))
                                  for r in rs if r["history"].get("fallbacks")],
                       equations=_stats(hist("equations")), fill_in=_stats(hist("fill_in")),
                       **{f"{k}_per_iteration": _stats(per_it(k))
                          for k in ("matrix_load_time", "matrix_factor_time", "matrix_solve_time")},
                       matrix_reorder_time=_stats(hist("matrix_reorder_time")), load_time=_stats(hist("load_time")))
        else:
            row.update(contraction_bound=_stats(hist("contraction_bound")), lnz=_stats(hist("lnz")),
                       symbolic_time=_stats(hist("symbolic_time")), numeric_time=_stats(hist("numeric_time")),
                       **{f"{k}_per_iteration": _stats(per_it(k))
                          for k in ("resolvent_time", "network_time", "triangular_solve_time", "check_time",
                                    "update_time")})
        rows.append(row)
    agree = agreement(out_dir, records, reference_dir)
    (out_dir / "summary.json").write_text(json.dumps(dict(rows=rows, agreement=agree), indent=2))

    med = lambda st, fmt: "-" if st is None else format(st["median"], fmt)
    rng = lambda st, fmt: "-" if st is None else (format(st["median"], fmt) if st["min"] == st["max"] else
                                                  f"{format(st['median'], fmt)} [{format(st['min'], fmt)}-"
                                                  f"{format(st['max'], fmt)}]")
    lines = [f"# Network models sweep: {out_dir.name}", "",
             f"Machine {spec.get('machine')}, started {spec.get('started')}. Medians over converged runs, "
             "[range] where seeds differ. Runtime is the whole solve; one-time and per-iteration costs as in "
             "network_models_experiment.run_costs. Memory is the run process's peak working set.", ""]
    for case in CASES:
        case_rows = [r for r in rows if r["case"] == case]
        if not case_rows:
            continue
        lines += [f"## {case}", "",
                  "| solver | size | ok/runs | runtime (s) | iterations | one-time (s) | per iteration (ms) | memory (GB) "
                  "| cpu/wall | KCL rel | notes |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in case_rows:
            notes = []
            if is_spice(r["algorithm"]):
                if r["methods"] != ["newton"]:
                    notes.append("method " + "/".join(r["methods"]))
                if r["fallbacks"]:
                    notes.append(f"fallbacks in {len(r['fallbacks'])} runs")
            if r["failures"]:
                notes.append("; ".join(f"seed {f['seed']} {f['status']}" + (f" ({f['method']})" if f["method"] else "")
                                       for f in r["failures"]))
            if r["skipped"]:
                notes.append(f"{r['skipped']} skipped")
            per = r["per_iteration"]
            lines.append(
                f"| {r['algorithm']} | {r['p']}x{r['q']} | {r['converged']}/{r['runs']} | {rng(r['runtime'], '.3g')} "
                f"| {rng(r['iterations'], '.0f')} | {med(r['one_time'], '.3g')} | "
                f"{'-' if per is None else format(1e3 * per['median'], '.3g')} | {med(r['peak_memory_gb'], '.2f')} | "
                f"{med(r['cpu_over_wall'], '.2f')} | {med(r['kcl_residual_rel'], '.1e')} | {'; '.join(notes)} |")
        lines.append("")
    skipped_only = sorted({(s["case"], s["algorithm"], s["p"], s["q"], s["reason"]) for s in skipped
                           if (s["case"], s["algorithm"], s["p"], s["q"]) not in groups})
    if skipped_only:
        lines += ["## Skipped", ""] + [f"- {c} {a} {p}x{q}: {why}" for c, a, p, q, why in skipped_only] + [""]
    if agree:
        lines += ["## Agreement", "", "Max |difference| of word and bit node potentials from the reference solution "
                  "(SPICE where it converged, spice_exp first, else spingarn_zero_bias), worst over seeds.", "",
                  "| case / solver / size | reference | max dV (V) |", "|---|---|---|"]
        for key, by_seed in agree.items():
            worst = max(by_seed.values(), key=lambda e: e["max_du"])
            lines.append(f"| {key} | {worst['reference']} | {worst['max_du']:.1e} |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    return rows


CSV_CASES = {"memristor": "mem", "1d1r_lumped": "lumped", "1d1r_separate": "separate",
             "1d1r_shockley_separate": "shockley"}
CSV_SOLVERS = {"spice": "spice", "spice_exp": "spice_exp", "spingarn_zero_bias": "zb", "spingarn_sector": "sector"}


def export(out_dir):
    """
    CSV files for plotting outside Python (e.g. pgfplots), from runs.jsonl and skipped.jsonl:

        runtime.csv  one row per array size N (all sizes are N x N), and for each case and solver in the sweep, as
                     <case>_<solver> with the short names of CSV_CASES and short_name, the median runtime in seconds over
                     converged runs, its _min, _max and _n (converged runs); nan where none converged;
        runs.csv     every run, one per line, skipped runs included with status "skipped" and their reason.
    """
    import csv

    out_dir = Path(out_dir)
    records = [json.loads(line) for line in open(out_dir / "runs.jsonl") if line.strip()]
    skipped = ([json.loads(line) for line in open(out_dir / "skipped.jsonl") if line.strip()]
               if (out_dir / "skipped.jsonl").exists() else [])
    times = defaultdict(list)
    for r in records:
        if r["status"] == CONVERGED:
            times[(r["case"], r["algorithm"], r["p"])].append(r["runtime"])
    sizes = sorted({r["p"] for r in records})
    cases = [c for c in CASES if any(r["case"] == c for r in records)]
    algorithms = sorted({r["algorithm"] for r in records}, key=solver_order)
    columns = ["N"] + [f"{CSV_CASES[c]}_{short_name(a)}{suffix}" for c in cases for a in algorithms
                       for suffix in ("", "_min", "_max", "_n")]
    with open(out_dir / "runtime.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for n in sizes:
            row = [n]
            for c in cases:
                for a in algorithms:
                    ts = times.get((c, a, n), [])
                    row += ([f"{statistics.median(ts):.6g}", f"{min(ts):.6g}", f"{max(ts):.6g}", len(ts)] if ts
                            else ["nan", "nan", "nan", 0])
            w.writerow(row)

    fields = ["case", "solver", "N", "seed", "status", "runtime_s", "iterations", "one_time_s", "iterations_total_s",
              "per_iteration_ms", "peak_memory_gb", "cpu_over_wall", "spice_method", "spice_fallbacks", "reason"]
    num = lambda x, scale=1.0: "" if x is None else f"{x * scale:.6g}"
    with open(out_dir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            h, c = r.get("history") or {}, r["costs"]
            w.writerow(dict(case=r["case"], solver=r["algorithm"], N=r["p"], seed=r["seed"], status=r["status"],
                            runtime_s=num(r["runtime"]),
                            iterations="" if c["iterations_run"] is None else int(c["iterations_run"]),
                            one_time_s=num(c["one_time"]), iterations_total_s=num(c["iterations_total"]),
                            per_iteration_ms=num(c["per_iteration"], 1e3),
                            peak_memory_gb=num(r.get("peak_memory"), 1e-9),
                            cpu_over_wall=num(h["cpu_time"] / r["runtime"]) if h.get("cpu_time") is not None
                            and r["runtime"] else "",
                            spice_method=h.get("method") or "",
                            spice_fallbacks="; ".join(f"{s} {o}" for s, o in h.get("fallbacks", [])),
                            reason=r.get("error") or ""))
        for s in skipped:
            w.writerow(dict(case=s["case"], solver=s["algorithm"], N=s["p"], seed=s["seed"], status="skipped",
                            reason=s["reason"]))
    return out_dir / "runtime.csv", out_dir / "runs.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="command", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--sizes", nargs="+", type=parse_size,
                    default=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
    sw.add_argument("--seeds", type=int, default=3, help="seeds 0..SEEDS-1 per size")
    sw.add_argument("--seeds-at", nargs="*", type=parse_seeds_at, default=[], metavar="SIZE=N",
                    help="a different number of seeds at a size, e.g. 1024=1")
    sw.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    sw.add_argument("--algorithms", nargs="+", choices=list(SOLVER_NAMES), default=list(DEFAULT_ALGORITHMS))
    sw.add_argument("--diode-gamma", type=lambda s: s if s in DIODE_GAMMA_RULES else float(s), default="r_min",
                    help="port rule of separate diodes (diode_port_resistances); not r_min adds a name suffix")
    sw.add_argument("--reference-dir", type=Path,
                    help="another sweep of the same circuits whose SPICE solutions the agreement check may use")
    sw.add_argument("--timeout", type=float, default=6 * 3600, help="seconds before a run is killed")
    sw.add_argument("--runtime-cap", type=float,
                    help="seconds into its timed solve before a run is killed and recorded as a timeout")
    sw.add_argument("--skip-seeds-after-timeout", nargs="*", metavar="SOLVER",
                    help="after a timeout of these solvers (full names; all if none given), skip their other seeds at "
                         "that size too, not only larger sizes")
    sw.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help="Spingarn's iteration cap")
    sw.add_argument("--spice-max-size", type=parse_size, help="skip SPICE above this size")
    sw.add_argument("--runtime-limit-at", nargs="*", type=parse_limit_at, default=[], metavar="SIZE=SECONDS",
                    help="at SIZE, skip each solver and case whose runtime, extrapolated from the two largest smaller "
                         "sizes (projected_runtime), exceeds SECONDS, e.g. 1024=3600")
    sw.add_argument("--no-skip", action="store_true", help="keep running a solver on a case after it failed there")
    sw.add_argument("--out", type=Path)
    sw.add_argument("--resume", action="store_true", help="continue the sweep in --out with its recorded settings")
    sm = sub.add_parser("summarize")
    sm.add_argument("dir", type=Path)
    sm.add_argument("--reference-dir", type=Path, help="as for sweep (default: the sweep's own)")
    ex = sub.add_parser("export", help="write runtime.csv and runs.csv (see export)")
    ex.add_argument("dir", type=Path)
    args = ap.parse_args()
    if args.command == "summarize":
        summarize(args.dir, args.reference_dir)
        print((args.dir / "summary.md").read_text())
    elif args.command == "export":
        print("\n".join(str(path) for path in export(args.dir)))
    else:
        if args.resume and args.out is None:
            ap.error("--resume needs --out")
        sweep(args)


if __name__ == "__main__":
    main()
