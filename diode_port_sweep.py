"""
Diode port sweep: how the port resistance of the ideal diodes on branches of their own (Config's internal_node, the
1d1r_separate case of network_models_experiment.py) sets fixed-Gamma Spingarn's iteration count. An ideal diode reflects
fully whatever its port (b = -|a|), so its port only changes how the network solve sees it: as gamma_d in series with the
memristor's port, whether the diode conducts or blocks.

Same circuit and solver settings as network_models_experiment.py's 1d1r_separate case (alpha 0.8, CHOLMOD in its own
ordering, fixed Gamma), with the memristors matched at zero bias ("r_lo") or over their sector ("geometric") and the
diode port by diode_port_resistances: a rule ("r_min", "geometric": per cell) or a resistance in ohms for every diode.
Each run is isolated (run_experiments.run_isolated); records go to runs.jsonl as each finishes, and --resume skips runs
already recorded.

    python diode_port_sweep.py --size 256 --ports r_min geometric 300 1e3 3e3 1e4 3e4 1e5 3e5 1e6 --seeds 1
"""
import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

from algorithms.spingarns import DIODE_GAMMA_RULES, SpingarnSolver
from network_models_experiment import ALPHA, CASES, named, run_costs
from run_experiments import git_commit, parse_size, run_isolated

ROOT = Path(__file__).resolve().parent
CASE = "1d1r_separate"
MATCHINGS = {"zero_bias": "r_lo", "sector": "geometric"}


def parse_port(text):
    return text if text in DIODE_GAMMA_RULES else float(text)


def solver(matching, port, max_iterations):
    s = SpingarnSolver(gamma=MATCHINGS[matching], diode_gamma=port, alpha=ALPHA[CASE], ordering="builtin",
                       max_iterations=max_iterations)
    return named(s, f"spingarn_{matching}_d{port if isinstance(port, str) else f'{port:g}'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--size", type=parse_size, default=(256, 256))
    ap.add_argument("--ports", nargs="+", type=parse_port, required=True)
    ap.add_argument("--matchings", nargs="+", choices=list(MATCHINGS), default=list(MATCHINGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "diode-port-256")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    p, q = args.size
    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "runs.jsonl"
    records = [json.loads(line) for line in open(runs_path) if line.strip()] if runs_path.exists() else []
    if records and not args.resume:
        raise SystemExit(f"{args.out} already holds run records; pass --resume to add to it")
    done = {(r["matching"], str(r["diode_port"]), r["p"], r["q"], r["seed"]) for r in records}
    spec_path = args.out / "experiment.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else dict(
        started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(), case=CASE,
        circuit=CASES[CASE], alpha=ALPHA[CASE], git_commit=git_commit(), invocations=[])
    spec["invocations"].append(dict(at=datetime.now().isoformat(timespec="seconds"), size=[p, q],
                                    ports=args.ports, matchings=args.matchings, seeds=args.seeds,
                                    max_iterations=args.max_iterations, timeout=args.timeout))
    spec_path.write_text(json.dumps(spec, indent=2))

    for seed in args.seeds:
        for matching in args.matchings:
            for port in args.ports:
                if (matching, str(port), p, q, seed) in done:
                    continue
                s = solver(matching, port, args.max_iterations)
                rec = run_isolated(s, p, q, seed, CASES[CASE], None, args.timeout)
                rec.update(case=CASE, matching=matching, diode_port=port, machine=platform.node(),
                           costs=run_costs(rec))
                with open(runs_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                c = rec["costs"]
                runtime = "-" if rec["runtime"] is None else f"{rec['runtime']:8.2f}s"
                per = "-" if c["per_iteration"] is None else f"{1e3 * c['per_iteration']:.1f} ms/it"
                print(f"{p}x{q} seed={seed} memristor {matching:<9} diode port {str(port):<9}: {rec['status']:<13} "
                      f"{runtime} its={c['iterations_run']} ({per})" + (f" ({rec['error']})" if rec["error"] else ""),
                      flush=True)
    print(f"results in {args.out}")


if __name__ == "__main__":
    main()
