"""
One single-axes figure per case for best_comparisons.md (report_assets/best_comparisons/), and a summary table per case on
stdout. Every case comes from one runner sweep on one machine, so its solvers are compared within that sweep only:

    results/best-memristor         memristor-only crossbar (report.md's nominal circuit)
    results/best-1s1r-Roff1M       1S1R, R_off 1 Mohm, kink (delta = 0)
    results/best-1d1r              ideal lumped 1D1R
    results/best-1d1r-internal     ideal diode on its own branch (internal node)
    results/diode-models-scale-Is1e-9   Shockley diode on its own branch (case "internal" of that sweep)

    python -X utf8 best_comparisons_figures.py
"""
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_assets" / "best_comparisons"
BLUE, ORANGE, TEAL, VIOLET = SERIES_COLORS[0], SERIES_COLORS[1], SERIES_COLORS[2], SERIES_COLORS[6]

# label, colour, line style (blue/orange/teal/violet: validator pass on all pairs, 2026-10-14)
STYLE = {
    "spingarn": ("Spingarn, fixed γ = R_min", BLUE, "-"),
    "spingarn_fixed": ("Spingarn, fixed γ = R_min", BLUE, "-"),
    "spingarn_sector": ("Spingarn, fixed sector matching", BLUE, "--"),
    "spingarn_adaptive": ("Spingarn, adaptive matching", VIOLET, "-"),
    "spingarn_adaptive_split": ("Spingarn, adaptive split", VIOLET, "--"),
    "newton": ("Newton (our CHOLMOD), plain", TEAL, "--"),
    "newton_safeguarded": ("Newton (our CHOLMOD), step halving", TEAL, "-"),
    "spice": ("SPICE (ngspice KLU)", ORANGE, "-"),
    "spice_exp": ("SPICE, behavioural exp diode", ORANGE, "--"),
    "spice_ron0.001": ("SPICE (ngspice KLU), R_on = 1 mΩ", ORANGE, "-"),
}

CASES = {
    "memristor": dict(run_dir=RESULTS / "best-memristor", case=None,
                      algorithms=["spingarn", "spingarn_adaptive", "spingarn_sector", "newton_safeguarded", "newton",
                                  "spice"],
                      title="Memristor-only crossbar: total runtime"),
    "1s1r": dict(run_dir=RESULTS / "best-1s1r-Roff1M", case=None,
                 algorithms=["spingarn_sector", "spingarn_adaptive", "newton_safeguarded", "newton", "spice"],
                 title="1S1R crossbar (R_off = 1 MΩ, kink): total runtime"),
    "1d1r_internal": dict(run_dir=RESULTS / "best-1d1r-internal", case=None,
                          algorithms=["spingarn", "spingarn_adaptive", "spice"],
                          title="Ideal diode on its own branch (exact for Spingarn): total runtime"),
    "1d1r": dict(run_dir=RESULTS / "best-1d1r", case=None,
                 algorithms=["spingarn", "spingarn_adaptive", "spingarn_adaptive_split", "newton_safeguarded", "newton",
                             "spice"],
                 title="Ideal lumped 1D1R crossbar: total runtime", max_size=256),
    # the same sweep with 512x512 (adaptive, split, both Newtons) and 1024x1024 (adaptive, split, step-halving Newton)
    # appended 2026-10-15 on the laptop; fixed and SPICE stop at 256. "1d1r" above stays capped at 256, as
    # best_comparisons.md shows it
    "1d1r_large": dict(run_dir=RESULTS / "best-1d1r", case=None,
                     algorithms=["spingarn", "spingarn_adaptive", "spingarn_adaptive_split", "newton_safeguarded",
                                 "newton", "spice"],
                     title="Ideal lumped 1D1R crossbar: total runtime, up to 1024×1024"),
    "shockley_internal": dict(run_dir=RESULTS / "diode-models-scale-Is1e-9", case="internal",
                              algorithms=["spingarn_fixed", "spingarn_adaptive", "newton_safeguarded", "newton", "spice",
                                          "spice_exp"],
                              title="Shockley diode (I_s = 1 nA) on its own branch: total runtime",
                              labels={"spice": "SPICE (ngspice KLU), D element"}),
    "shockley_internal_no_newton": dict(run_dir=RESULTS / "diode-models-scale-Is1e-9", case="internal",
                                        algorithms=["spingarn_fixed", "spingarn_adaptive", "spice", "spice_exp"],
                                        title="Shockley diode (I_s = 1 nA) on its own branch: total runtime",
                                        labels={"spice": "SPICE (ngspice KLU), D element"}),
}


def load(run_dir, case):
    with open(Path(run_dir) / "runs.jsonl") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    return [r for r in runs if case is None or r.get("case") == case]


def machine(run_dir):
    spec = json.loads((Path(run_dir) / "experiment.json").read_text())
    return spec.get("machine") or "ryan-work"


def grouped(runs):
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        out[r["algorithm"]][r["p"]].append(r)
    return out


def figure(key):
    spec = CASES[key]
    runs = [r for r in load(spec["run_dir"], spec["case"]) if r["p"] <= spec.get("max_size", r["p"])]
    groups = grouped(runs)
    sizes = sorted({r["p"] for r in runs})
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    fail_row = 0
    for alg in spec["algorithms"]:
        if alg not in groups:
            continue
        label, color, ls = STYLE[alg]
        label = spec.get("labels", {}).get(alg, label)
        xs, ys, fails = [], [], []
        for p in sizes:
            rs = groups[alg].get(p, [])
            ok = [r for r in rs if r["status"] == "converged"]
            if len(ok) < len(rs):
                fails.append(p * p)
            for r in ok:
                fallback = r["history"].get("method") not in (None, "newton")
                ax.plot(p * p, r["runtime"], "o", ms=11 if fallback else 3.5, mfc="none" if fallback else color,
                        mec=INK_2 if fallback else color, mew=1.1, alpha=0.9 if fallback else 0.5, zorder=4)
            if ok:
                xs.append(p * p)
                ys.append(np.median([r["runtime"] for r in ok]))
        top = max(groups[alg])  # the largest size this solver ran, which may be below the sweep's
        last = [r for r in groups[alg][top] if r["status"] == "converged"]
        its = f"{int(np.median([r['iterations'] for r in last]))} its at {top}×{top}" if last else \
            f"failed at {top}×{top}"
        if xs:
            ax.plot(xs, ys, color=color, ls=ls, lw=2.2, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3,
                    label=f"{label} ({its})")
        else:
            ax.plot([], [], color=color, ls=ls, lw=2.2, label=f"{label} ({its})")
        if fails:
            ax.plot(fails, [1.02 + 0.04 * fail_row] * len(fails), "x", ms=8, mew=2, color=color, clip_on=False,
                    transform=ax.get_xaxis_transform())
            fail_row += 1
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([p * p for p in sizes], [f"{p}×{p}" for p in sizes])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_title(spec["title"], loc="left", color=INK, fontsize=11.5, pad=24 if fail_row else 8)
    ax.set_xlabel("array size", color=INK_2)
    ax.set_ylabel("total runtime (s), median of 3 seeds", color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=3.2, loc="upper left")
    note = f"Measured on the laptop ({machine(spec['run_dir'])}), one sweep. Small dots: each seed"
    if any(r["history"].get("method") not in (None, "newton") for r in runs if r["status"] == "converged"):
        note += "; ringed: SPICE needed a fallback (gmin or source stepping)"
    if fail_row:
        note += "; × above the plot: a seed did not converge"
    note = textwrap.fill(note + ".", 150)
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.02 * (1 + note.count(chr(10))), 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{key}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    table(key, groups, sizes)


def table(key, groups, sizes):
    """Largest and second-largest size: iterations, runtime (median and range), peak memory, fallbacks, failures."""
    print(f"\n=== {key}")
    for p in sizes[-2:]:
        print(f"  {p}x{p}")
        for alg, by_size in groups.items():
            rs = by_size.get(p, [])
            ok = [r for r in rs if r["status"] == "converged"]
            if not rs:
                continue
            if not ok:
                print(f"    {alg:<26} failed {len(rs)}/{len(rs)}: {[r['history'].get('failure') or r['status'] for r in rs]}")
                continue
            rt = sorted(r["runtime"] for r in ok)
            its = sorted(r["iterations"] for r in ok)
            mem = [r["peak_memory"] / 1e9 for r in ok if r.get("peak_memory")]
            extra = ""
            if any("rematches" in r["history"] for r in ok):
                extra += f" rematches {sorted(r['history']['rematches'] for r in ok)}"
            if any("halvings" in r["history"] for r in ok):
                extra += f" halvings {sorted(r['history']['halvings'] for r in ok)}"
            if any("method" in r["history"] for r in ok):
                extra += f" methods {sorted(set(r['history']['method'] for r in ok))}"
            if any("kcl_residual" in r["history"] for r in ok):
                extra += f" kcl_abs<={max(r['history']['kcl_residual'] for r in ok):.1e}"
            failed = len(rs) - len(ok)
            print(f"    {alg:<26} its {its[0]}-{its[-1]}  runtime median {np.median(rt):.3g} s [{rt[0]:.3g}-{rt[-1]:.3g}]"
                  f"  mem {max(mem) if mem else float('nan'):.2f} GB{extra}" + (f"  FAILED {failed}" if failed else ""))


def main():
    for key, spec in CASES.items():
        if (spec["run_dir"] / "runs.jsonl").exists():
            figure(key)
        else:
            print(f"skipped {key}: no {spec['run_dir']}")


if __name__ == "__main__":
    main()
