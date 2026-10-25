"""
CSV files and LaTeX tables for the pgfplots figures of the SPICE vs Spingarn comparison on three crossbar models, merged
from the 10-seed network-models sweeps on the PC (network_models_experiment.py, results/network-models-10seeds/; the task
is in instructions.md, rerun 2026-10-24 with fresh data, 10 seeds and a 1600 s runtime cap):

    mem       the memristor alone, alpha 1: SPICE and Spingarn zero-bias (gamma R_min);
    ideal     1D1R with an ideal diode on a branch of its own, alpha 0.8: SPICE (uramp / 1 mohm diode), Spingarn with
              zero-bias memristors and the geometric diode port (diode_gamma "geometric");
    shockley  as ideal with a Shockley diode (I_s 1e-9 A, n 1), in SPICE a behavioural exp() source (spice_exp).

All Spingarn runs keep Gamma fixed and use CHOLMOD's own ordering (AMD). SERIES says which sweep folder each series
comes from. A run still solving 1600 s into its solve was killed and counts as "failed to converge within the time limit"
(a timeout). Written to latex/network_models/ (or --out), next to the .tex files that plot them:

    runtime.csv, iterations.csv
        one row per array size N (N x N), and per series: the median over the runs attempted at that size, timeouts
        counted as above the cap (a censored median: nan once half or more of the runs timed out); _min and _max over the
        converged runs, the runtime's _max raised to the cap where some run timed out; _n converged runs, _runs runs
        attempted, _timeout timeouts; and _capped, the cap where the median is nan because of timeouts (where the
        figures mark the time limit), else nan. Units: s, iterations (SPICE: ngspice's Newton iterations, fallback
        stages included).
    memory.csv, per_iteration.csv, one_time.csv
        per series the median over converged runs, _min, _max, _n; nan where none converged. Units: GB, ms, s.
    outcomes.csv
        per size and series: runs converged, timed out, failed otherwise (crash, iteration cap, SPICE failure), skipped.
    agreement.csv
        per size, each Spingarn series' max |difference| of word and bit node potentials (V) from SPICE's solution of
        the same circuit (spice_exp's on shockley), worst over seeds; nan where SPICE has no solution.
    runs.csv
        every run, one per line, with its series; skipped runs included with status "skipped" and the reason.
    table_runtime.tex, table_iterations.tex
        booktabs tables of the medians (runtime with the SPICE/Spingarn ratio), for \\input into a table environment.

    python network_models_latex.py [--results results] [--out latex/network_models]
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from Result import CONVERGED, TIMEOUT

ROOT = Path(__file__).resolve().parent
MACHINE = "DESKTOP-EFQTE92"  # the PC: every series must come from it (memory/measurement-machine.md)
SWEEPS = "network-models-10seeds"
CAP = 1600.0  # the runtime cap of the sweeps (s); checked against each sweep's experiment.json
# column -> (case, solver, sweep folders under results/)
SERIES = {
    "mem_spice": ("memristor", "spice", [f"{SWEEPS}/1-memristor"]),
    "mem_zb": ("memristor", "spingarn_zero_bias", [f"{SWEEPS}/1-memristor"]),
    "ideal_spice": ("1d1r_separate", "spice", [f"{SWEEPS}/2-ideal-separate"]),
    "ideal_zb": ("1d1r_separate", "spingarn_zero_bias_dgeometric", [f"{SWEEPS}/2-ideal-separate"]),
    "shockley_spice": ("1d1r_shockley_separate", "spice_exp", [f"{SWEEPS}/3-shockley-separate"]),
    "shockley_zb": ("1d1r_shockley_separate", "spingarn_zero_bias_dgeometric", [f"{SWEEPS}/3-shockley-separate"]),
}
CASES = {"mem": "Memristor", "ideal": "1D1R, ideal diode", "shockley": "1D1R, Shockley diode"}
METRICS = {
    "runtime": lambda r: r["runtime"],
    "iterations": lambda r: r["iterations"],
    "memory": lambda r: None if r.get("peak_memory") is None else r["peak_memory"] / 1e9,
    "per_iteration": lambda r: None if r["costs"]["per_iteration"] is None else 1e3 * r["costs"]["per_iteration"],
    "one_time": lambda r: r["costs"]["one_time"],
}
CENSORED = ("runtime", "iterations")  # metrics a timeout bounds from below


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()] if path.exists() else []


def timed_out(r):
    """Whether r failed to converge within the runtime cap (killed at the cap, or converged only after it)."""
    return r["status"] == TIMEOUT and (r.get("history") or {}).get("runtime_cap") is not None


def load(results):
    """Per series, its run records and skipped runs, and its agreement entries, from the folders in SERIES."""
    runs, skipped, agree = defaultdict(list), defaultdict(list), defaultdict(dict)
    for column, (case, solver, folders) in SERIES.items():
        seen = set()
        reference = SERIES[column.split("_")[0] + "_spice"][1]  # the SPICE solver of this series' case
        for folder in folders:
            d = results / folder
            spec = json.loads((d / "experiment.json").read_text()) if (d / "experiment.json").exists() else {}
            if spec and spec.get("runtime_cap") != CAP:
                raise SystemExit(f"{d}: runtime cap {spec.get('runtime_cap')}, not {CAP:g} s")
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


def stats(runs, n, metric):
    """The CSV entries of one series at size N for a metric (see the module docstring), as a dict of numbers/None."""
    attempted = [r for r in runs if r["p"] == n]
    ok = [x for x in (METRICS[metric](r) for r in attempted if r["status"] == CONVERGED) if x is not None]
    timeouts = sum(map(timed_out, attempted))
    out = dict(median=None, min=min(ok, default=None), max=max(ok, default=None), n=len(ok), runs=len(attempted),
               timeout=timeouts, capped=None)
    if metric in CENSORED:
        values = ok + [math.inf] * timeouts
        median = statistics.median(values) if values else None
        out["median"] = None if median is None or math.isinf(median) else median
        if timeouts:
            out["max"] = CAP if metric == "runtime" else out["max"]
            out["capped"] = CAP if out["median"] is None else None
    else:
        out["median"] = statistics.median(ok) if ok else None
    return out


def fmt(x):
    return "nan" if x is None else f"{x:.6g}"


def tex_number(x, digits=3):
    """x to `digits` significant figures, without exponent notation, for the runtimes in the tables."""
    if x >= 10 ** (digits - 1):
        return f"{x:.0f}"
    return f"{x:.{digits - 1 - math.floor(math.log10(abs(x)))}f}"


def write_tables(table, out, sizes, skipped):
    """table_runtime.tex and table_iterations.tex from the medians (table[metric][column][n] = stats(...))."""

    def cell(column, n, metric):
        st = table[metric][column].get(n)
        if st is None or st["runs"] == 0:
            return "--"
        if st["median"] is None:
            return f"$>${CAP:.0f}" if metric == "runtime" and st["timeout"] else ("t/o" if st["timeout"] else "fail")
        text = tex_number(st["median"]) if metric == "runtime" else f"{st['median']:.0f}"
        if st["n"] < st["runs"]:
            text += f"\\textsuperscript{{{st['n']}/{st['runs']}}}"
        return text

    def ratio(case, n):
        spice, zb = table["runtime"][f"{case}_spice"].get(n), table["runtime"][f"{case}_zb"].get(n)
        if not spice or not zb or zb["median"] is None or spice["runs"] == 0:
            return "--"
        if spice["median"] is None:
            return f"$>${CAP / zb['median']:.1f}" if spice["timeout"] else "--"
        return f"{spice['median'] / zb['median']:.1f}"

    def header(k, names):
        return (f"\\begin{{tabular}}{{r{'r' * k * len(CASES)}}}\n\\toprule\n"
                + " & ".join([""] + [f"\\multicolumn{{{k}}}{{c}}{{{title}}}" for title in CASES.values()]) + r" \\"
                + "\n" + "".join(f"\\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(len(CASES)))
                + "\n" + " & ".join(["$N$"] + names * len(CASES)) + " \\\\\n\\midrule")

    lines = [header(3, ["SPICE", "Spingarn", "ratio"])]
    for n in sizes:
        row = [str(n)]
        for case in CASES:
            row += [cell(f"{case}_spice", n, "runtime"), cell(f"{case}_zb", n, "runtime"), ratio(case, n)]
        lines.append(" & ".join(row) + r" \\")
    lines.append("\\bottomrule\n\\end{tabular}")
    (out / "table_runtime.tex").write_text(
        "% Median runtime (s) over 10 seeds, the PC (DESKTOP-EFQTE92); written by network_models_latex.py.\n"
        "% ratio = SPICE / Spingarn (> 1: Spingarn faster). Superscript k/m: only k of m runs attempted converged (the\n"
        f"% median counts timed-out runs as above the {CAP:g} s cap and leaves other failures out). >{CAP:.0f} s: half or\n"
        "% more of the runs hit the cap; --: not run (SPICE at 1024: ngspice's KLU size limit). Needs booktabs.\n"
        + "\n".join(lines) + "\n")

    lines = [header(2, ["SPICE", "Spingarn"])]
    for n in sizes:
        row = [str(n)]
        for case in CASES:
            row += [cell(f"{case}_spice", n, "iterations"), cell(f"{case}_zb", n, "iterations")]
        lines.append(" & ".join(row) + r" \\")
    lines.append("\\bottomrule\n\\end{tabular}")
    (out / "table_iterations.tex").write_text(
        "% Median iterations over 10 seeds (SPICE: ngspice's Newton iterations, fallback stages included); written by\n"
        "% network_models_latex.py. t/o: half or more of the runs hit the cap; superscripts and -- as in\n"
        "% table_runtime.tex. Needs booktabs.\n" + "\n".join(lines) + "\n")


def write(results, out):
    runs, skipped, agree = load(results)
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({r["p"] for rs in runs.values() for r in rs} | {s["p"] for ss in skipped.values() for s in ss})
    table = {}
    for metric in METRICS:
        table[metric] = {c: {n: stats(runs[c], n, metric) for n in sizes} for c in SERIES}
        suffixes = (("", "median"), ("_min", "min"), ("_max", "max"), ("_n", "n"))
        if metric in CENSORED:
            suffixes += (("_runs", "runs"), ("_timeout", "timeout"), ("_capped", "capped"))
        with open(out / f"{metric}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N"] + [f"{c}{s}" for c in SERIES for s, _ in suffixes])
            for n in sizes:
                row = [n]
                for c in SERIES:
                    st = table[metric][c][n]
                    row += [st[k] if k in ("n", "runs", "timeout") else fmt(st[k]) for _, k in suffixes]
                w.writerow(row)

    with open(out / "outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N"] + [f"{c}_{k}" for c in SERIES for k in ("converged", "timeout", "failed", "skipped")])
        for n in sizes:
            row = [n]
            for c in SERIES:
                rs = [r for r in runs[c] if r["p"] == n]
                conv, tout = sum(r["status"] == CONVERGED for r in rs), sum(map(timed_out, rs))
                row += [conv, tout, len(rs) - conv - tout, sum(s["p"] == n for s in skipped[c])]
            w.writerow(row)

    zb = [c for c in SERIES if not c.endswith("_spice")]
    with open(out / "agreement.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N"] + zb)
        for n in sizes:
            w.writerow([n] + [fmt(agree[c].get(n)) for c in zb])

    fields = ["series", "case", "solver", "N", "seed", "status", "runtime_s", "iterations", "one_time_s",
              "per_iteration_ms", "peak_memory_gb", "spice_method", "spice_fallbacks", "kcl_residual_rel", "reason"]
    num = lambda x, scale=1.0: "" if x is None else fmt(x * scale)
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for column, (case, solver, _) in SERIES.items():
            for r in sorted(runs[column], key=lambda r: (r["p"], r["seed"])):
                h, c = r.get("history") or {}, r["costs"]
                w.writerow(dict(series=column, case=case, solver=solver, N=r["p"], seed=r["seed"],
                                status="timeout (over the runtime cap)" if timed_out(r) else r["status"],
                                runtime_s=num(r["runtime"]),
                                iterations="" if c["iterations_run"] is None else int(c["iterations_run"]),
                                one_time_s=num(c["one_time"]), per_iteration_ms=num(c["per_iteration"], 1e3),
                                peak_memory_gb=num(r.get("peak_memory"), 1e-9),
                                spice_method=(h.get("method") or "") if solver.startswith("spice") else "",
                                spice_fallbacks="; ".join(f"{s} {o}" for s, o in h.get("fallbacks", [])),
                                kcl_residual_rel=num(h.get("kcl_residual_rel")),
                                reason=r.get("error") or ""))
            for s in sorted(skipped[column], key=lambda s: (s["p"], s["seed"])):
                w.writerow(dict(series=column, case=case, solver=solver, N=s["p"], seed=s["seed"], status="skipped",
                                reason=s["reason"]))
    write_tables(table, out, sizes, skipped)
    for column, rs in runs.items():
        ok = sorted({r["p"] for r in rs if r["status"] == CONVERGED})
        bad = [(r["p"], r["seed"], r["status"]) for r in rs if r["status"] != CONVERGED]
        print(f"{column:<15} {len(rs):3d} runs, converged at N = {ok}" + (f"; not converged: {bad}" if bad else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "latex" / "network_models")
    args = ap.parse_args()
    print(write(args.results, args.out))


if __name__ == "__main__":
    main()
