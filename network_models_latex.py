"""
CSV files for the LaTeX (pgfplots) figures of the SPICE vs Spingarn comparison on three crossbar models, merged from the
network-models sweeps on the PC (network_models_experiment.py; the task and its settings are in instructions.md):

    mem       the memristor alone, alpha 1: SPICE, Spingarn zero-bias (gamma R_min) and sector (sqrt(R_min R_max));
    ideal     1D1R with an ideal diode on a branch of its own, alpha 0.8: SPICE (uramp / 1 mohm diode), Spingarn with
              zero-bias memristors and the geometric diode port (diode_gamma "geometric");
    shockley  as ideal with a Shockley diode (I_s 1e-9 A, n 1), in SPICE a behavioural exp() source (spice_exp; the
              user's choice: the D element runs in the same folder, solver "spice", are left out).

All Spingarn runs keep Gamma fixed and use CHOLMOD's own ordering (AMD). SERIES says which sweep folder each series
comes from. Written to latex/network_models/ (or --out), next to the .tex files that plot them:

    runtime.csv, iterations.csv, memory.csv, per_iteration.csv, one_time.csv
        one row per array size N (N x N), and per series its median over converged runs, then _min, _max and _n
        (converged runs); nan where none converged (pgfplots leaves those points out). Units: s, iterations, GB, ms, s.
        SPICE's iterations are ngspice's Newton iterations, fallback stages included.
    agreement.csv
        per size, each Spingarn series' max |difference| of word and bit node potentials (V) from SPICE's solution of
        the same circuit (spice_exp's on shockley), worst over seeds; nan where SPICE has no solution.
    runs.csv
        every run, one per line, with its series; skipped runs included with status "skipped" and the reason.

    python network_models_latex.py [--results results] [--out latex/network_models]
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from Result import CONVERGED

ROOT = Path(__file__).resolve().parent
MACHINE = "DESKTOP-EFQTE92"  # the PC: every series must come from it (memory/measurement-machine.md)
# column -> (case, solver, sweep folders under results/)
SERIES = {
    "mem_spice": ("memristor", "spice", ["network-models-1"]),
    "mem_zb": ("memristor", "spingarn_zero_bias", ["network-models-1"]),
    "mem_sector": ("memristor", "spingarn_sector", ["network-models-1"]),
    "ideal_spice": ("1d1r_separate", "spice", ["network-models-1"]),
    "ideal_zb": ("1d1r_separate", "spingarn_zero_bias_dgeometric",
                 ["network-models-2-dgeometric", "network-models-2-dgeometric-1024"]),
    "shockley_spice": ("1d1r_shockley_separate", "spice_exp", ["network-models-3-shockley"]),
    "shockley_zb": ("1d1r_shockley_separate", "spingarn_zero_bias_dgeometric", ["network-models-3-shockley"]),
}
METRICS = {
    "runtime": lambda r: r["runtime"],
    "iterations": lambda r: r["iterations"],
    "memory": lambda r: None if r.get("peak_memory") is None else r["peak_memory"] / 1e9,
    "per_iteration": lambda r: None if r["costs"]["per_iteration"] is None else 1e3 * r["costs"]["per_iteration"],
    "one_time": lambda r: r["costs"]["one_time"],
}


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()] if path.exists() else []


def load(results):
    """Per series, its run records and skipped runs, and its agreement entries, from the folders in SERIES."""
    runs, skipped, agree = defaultdict(list), defaultdict(list), defaultdict(dict)
    for column, (case, solver, folders) in SERIES.items():
        seen = set()
        reference = SERIES[column.split("_")[0] + "_spice"][1]  # the SPICE solver of this series' case
        for folder in folders:
            d = results / folder
            for r in read_jsonl(d / "runs.jsonl"):
                if (r["case"], r["algorithm"]) != (case, solver):
                    continue
                if r["machine"] != MACHINE:
                    raise SystemExit(f"{d}: a {column} run is from {r['machine']}, not {MACHINE}")
                key = (r["p"], r["q"], r["seed"])
                if key in seen:
                    raise SystemExit(f"{column} {key} appears twice in {folders}")
                seen.add(key)
                runs[column].append(r)
            skipped[column] += [s for s in read_jsonl(d / "skipped.jsonl") if (s["case"], s["algorithm"]) == (case, solver)]
            summary = d / "summary.json"
            for key, by_seed in (json.loads(summary.read_text())["agreement"] if summary.exists() else {}).items():
                c, a, size = key.split("/")
                vs_spice = [e["max_du"] for e in by_seed.values() if e["reference"].split(" ")[0] == reference]
                if (c, a) == (case, solver) and vs_spice:
                    agree[column][int(size.split("x")[0])] = max(vs_spice)
        if not runs[column]:
            print(f"warning: no runs for {column} ({case}, {solver}) in {folders}; its columns are nan")
    return runs, skipped, agree


def write(results, out):
    runs, skipped, agree = load(results)
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({r["p"] for rs in runs.values() for r in rs})
    fmt = lambda x: f"{x:.6g}"
    for metric, value in METRICS.items():
        with open(out / f"{metric}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N"] + [f"{c}{s}" for c in SERIES for s in ("", "_min", "_max", "_n")])
            for n in sizes:
                row = [n]
                for column in SERIES:
                    xs = [value(r) for r in runs[column] if r["p"] == n and r["status"] == CONVERGED]
                    xs = [x for x in xs if x is not None]
                    row += [fmt(statistics.median(xs)), fmt(min(xs)), fmt(max(xs)), len(xs)] if xs else ["nan"] * 3 + [0]
                w.writerow(row)

    zb = [c for c in SERIES if not c.endswith("_spice")]
    with open(out / "agreement.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N"] + zb)
        for n in sizes:
            w.writerow([n] + [fmt(agree[c][n]) if n in agree[c] else "nan" for c in zb])

    fields = ["series", "case", "solver", "N", "seed", "status", "runtime_s", "iterations", "one_time_s",
              "per_iteration_ms", "peak_memory_gb", "spice_method", "spice_fallbacks", "reason"]
    num = lambda x, scale=1.0: "" if x is None else fmt(x * scale)
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for column, (case, solver, _) in SERIES.items():
            for r in sorted(runs[column], key=lambda r: (r["p"], r["seed"])):
                h, c = r.get("history") or {}, r["costs"]
                w.writerow(dict(series=column, case=case, solver=solver, N=r["p"], seed=r["seed"], status=r["status"],
                                runtime_s=num(r["runtime"]),
                                iterations="" if c["iterations_run"] is None else int(c["iterations_run"]),
                                one_time_s=num(c["one_time"]), per_iteration_ms=num(c["per_iteration"], 1e3),
                                peak_memory_gb=num(r.get("peak_memory"), 1e-9),
                                spice_method=(h.get("method") or "") if solver == "spice" else "",
                                spice_fallbacks="; ".join(f"{s} {o}" for s, o in h.get("fallbacks", [])),
                                reason=r.get("error") or ""))
            for s in skipped[column]:
                w.writerow(dict(series=column, case=case, solver=solver, N=s["p"], seed=s["seed"], status="skipped",
                                reason=s["reason"]))
    for column, rs in runs.items():
        ok = sorted({r["p"] for r in rs if r["status"] == CONVERGED})
        print(f"{column:<15} {len(rs):3d} runs, converged at N = {ok}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "latex" / "network_models")
    args = ap.parse_args()
    print(write(args.results, args.out))


if __name__ == "__main__":
    main()
