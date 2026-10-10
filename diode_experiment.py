"""
Spingarn's iteration on a 1D1R crossbar (an ideal diode in series with every memristor, no leak): iterations and
convergence rate vs array size for several relaxations alpha, the linearized rate at the solution, and how much of the
devices' nonlinearity the operating point exercises.

Without a leak the 1D1R law is flat in reverse bias, so the a priori contraction bound is 1: convergence is guaranteed
only for alpha < 1 (Krasnosel'skii-Mann), with no rate. The linearized iteration at the solution is
J = (1 - alpha) I + alpha S D with D = diag of the branch reflection factors; with zero-bias matching (gamma = R_min per
cell) D >= 0, so S D is similar to the symmetric D^1/2 S~ D^1/2 (S~ = Gamma^-1/2 S Gamma^1/2, an orthogonal reflection)
and has real eigenvalues in [-1, 1]. Its extreme eigenvalues lam_min, lam_max, found by Lanczos without forming any
matrix, give rho(J) = max |1 - alpha + alpha lam| over the two, for every alpha at once.

Nonlinearity diagnostics at the solution, per cell: whether its diode blocks (v <= 0), and for conducting cells the
chord deficit 1 - (i / v) R_min, how far the memristor's chord conductance has fallen below its zero-bias value (0 for a
linear device, ~v^2 / 3 for small v). Two output checks solve the same circuit with one ingredient removed and report the
relative change of the bit-line output currents (through the loads): memristors made linear at R_min (the tanh's
effect) and diodes removed (the diodes' effect).

Every runtime here is a single in-process timing; it is recorded for orientation, not for comparing solvers.

    python diode_experiment.py --sizes 8 16 32 64 128 256 512 1024 --seeds 3
    python diode_experiment.py --replot results/diode-20261009-120000

Writes results/diode-<timestamp>/ (or --out): experiment.json, records.jsonl (one record per size and seed, appended as
each finishes) and diode_experiment.png.
"""
import argparse
import copy
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Config import BranchKind, Config
from algorithms.ordering import nested_dissection
from algorithms.spingarns import SpingarnSolver, blas_threads, factorize, port_resistances, reflection
from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE
from run_experiments import git_commit, parse_size

ROOT = Path(__file__).resolve().parent
CIRCUIT = dict(device="1d1r", E_range=(-1.0, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
TOL = 1e-9


def variant(cfg, linear=False, diode=True):
    """cfg with its memristors made linear at their zero-bias resistance (i = v / R_min) and/or its diodes removed;
    the same sources, wires and cells otherwise."""
    out = copy.copy(cfg)
    out.r_hi = cfg.r_hi.copy()
    d = cfg.slices[BranchKind.DEVICE]
    if linear:
        out.R_max_cell = cfg.R_min_cell.copy()  # the tanh term's coefficient 1/R_min - 1/R_max vanishes
    if not diode:
        out.device = "memristor"
    out.r_hi[d] = out.R_max_cell if out.device == "memristor" else np.inf
    return out


def operating_point(cfg, v, i):
    """How the cells sit at the solution v, i (see the module docstring)."""
    d = cfg.slices[BranchKind.DEVICE]
    x, cur = v[d], i[d]
    fwd = x > 0
    deficit = 1 - cur[fwd] / x[fwd] * cfg.R_min_cell[fwd]
    total = np.abs(cur).sum()
    share = lambda mask: float(np.abs(cur[fwd][mask]).sum() / total) if total > 0 else 0.0
    q = lambda arr, p: float(np.quantile(arr, p)) if arr.size else None
    return dict(
        reverse=float(np.mean(~fwd)),
        near_kink=float(np.mean(np.abs(x) < 1e-3)),
        v_fwd_median=q(x[fwd], 0.5), v_fwd_p90=q(x[fwd], 0.9), v_fwd_max=q(x[fwd], 1.0),
        v_rev_median=q(x[~fwd], 0.5),
        deficit_median=q(deficit, 0.5), deficit_p90=q(deficit, 0.9), deficit_max=q(deficit, 1.0),
        cells_deficit_over_10pct=float(np.sum(deficit > 0.1) / x.size),
        current_share_deficit_over_10pct=share(deficit > 0.1),
        current_share_deficit_over_1pct=share(deficit > 0.01),
    )


def output_change(cfg, ref, alt):
    """Relative change of the bit-line output currents (through the loads) from solution ref to solution alt."""
    load = cfg.slices[BranchKind.LOAD]
    return float(np.linalg.norm(alt[load] - ref[load]) / np.linalg.norm(ref[load]))


def linearized_spectrum(cfg, gamma, v):
    """Extreme eigenvalues of S D at the solution v, by Lanczos on the symmetric D^1/2 S~ D^1/2 (needs D >= 0)."""
    A = cfg.incidence()
    AT = A.T.tocsr()
    with blas_threads(1):
        solve, _ = factorize(A @ sp.diags(1 / gamma) @ AT, "cholmod", nested_dissection(cfg))
    with np.errstate(divide="ignore"):
        R = 1 / cfg.conductance(v)  # inf on reverse-biased cells
    D = reflection(R, gamma)
    assert D.min() > -1e-12, "S D is only symmetrizable for D >= 0, i.e. zero-bias matching"
    s, g = np.sqrt(np.clip(D, 0, None)), np.sqrt(gamma)

    def matvec(x):
        z = g * (s * np.ravel(x))  # scaled waves -> waves
        return s * ((2 * (AT @ solve(A @ (z / gamma))) - z) / g)

    op = spla.LinearOperator((cfg.num_branches,) * 2, matvec=matvec, dtype=float)
    out = {}
    for which in ("SA", "LA"):
        out[which] = float(spla.eigsh(op, k=1, which=which, ncv=40, tol=1e-10, maxiter=5000,
                                      return_eigenvectors=False)[0])
    return out["SA"], out["LA"], int(np.sum(D == 1.0))


def tail_rate(history):
    """Observed per-iteration rate of the fixed-point residual over the second half of its history, above the
    rounding floor."""
    r = np.asarray(history)
    r = r[: max(np.flatnonzero(r > 1e-14)[-1] + 1, 2)] if np.any(r > 1e-14) else r
    k0 = len(r) // 2
    return float((r[-1] / r[k0]) ** (1 / (len(r) - 1 - k0))) if len(r) - 1 - k0 > 0 else None


def run_case(p, q, seed, circuit, alphas, max_iterations, spectrum):
    cfg = Config(p, q, seed=seed, **circuit)
    record = dict(p=p, q=q, seed=seed, config=cfg.params(), solves={})
    best = None
    for alpha in alphas:
        solver = SpingarnSolver(tol=TOL, max_iterations=max_iterations, alpha=alpha)
        t = time.perf_counter()
        out = solver.solve(cfg)
        runtime = time.perf_counter() - t
        h = out.history
        record["solves"][str(alpha)] = dict(
            converged=out.converged, iterations=out.iterations, runtime=runtime,
            iteration_time=h["iteration_time"], setup_time=h["setup_time"],
            kcl_residual_rel=h["kcl_residual_rel"], kvl_residual_rel=h["kvl_residual_rel"],
            fixed_point_final=h["fixed_point_history"][-1], observed_rate=tail_rate(h["fixed_point_history"]),
            newton_steps=h["newton_steps"])
        if out.converged and (best is None or out.iterations < best[1].iterations):
            best = (alpha, out)
    if best is None:
        record["error"] = "no alpha converged"
        return record
    alpha_ref, ref = best
    record["reference_alpha"] = alpha_ref
    record["operating_point"] = operating_point(cfg, ref.v, ref.i)

    # what each ingredient does to the outputs: same circuit without the tanh, and without the diodes
    for name, alt_cfg, alt_alpha in [("linear_memristor", variant(cfg, linear=True), alpha_ref),
                                     ("no_diode", variant(cfg, diode=False), 1.0)]:
        alt = SpingarnSolver(tol=TOL, max_iterations=max(max_iterations, 2000), alpha=alt_alpha).solve(alt_cfg)
        record[f"output_change_{name}"] = output_change(cfg, ref.i, alt.i) if alt.converged else None

    if spectrum:
        t = time.perf_counter()
        lam_min, lam_max, fully_reflecting = linearized_spectrum(cfg, port_resistances(cfg, "r_lo"), ref.v)
        record["spectrum"] = dict(lam_min=lam_min, lam_max=lam_max, fully_reflecting_cells=fully_reflecting,
                                  time=time.perf_counter() - t)
        for alpha in alphas:
            rho = max(abs(1 - alpha + alpha * lam_min), abs(1 - alpha + alpha * lam_max))
            record["solves"][str(alpha)]["linearized_rate"] = rho
            record["solves"][str(alpha)]["predicted_iterations"] = float(np.log(TOL) / np.log(rho)) if rho < 1 else None
        record["spectrum"]["best_alpha"] = 2 / (2 - lam_min - lam_max)
        record["spectrum"]["best_rate"] = 1 - record["spectrum"]["best_alpha"] * (1 - lam_max)
    return record


def report(rec, alphas):
    size = f"{rec['p']}x{rec['q']} seed {rec['seed']}"
    if "error" in rec:
        print(f"{size}: {rec['error']}", flush=True)
        return
    parts = []
    for alpha in alphas:
        s = rec["solves"][str(alpha)]
        its = f"{s['iterations']}" if s["converged"] else f">{s['iterations']}"
        lin = f" rho {s['linearized_rate']:.4f}" if "linearized_rate" in s else ""
        obs = f" obs {s['observed_rate']:.4f}" if s["observed_rate"] is not None else ""
        parts.append(f"a={alpha}: {its:>5} its{lin}{obs}")
    op = rec["operating_point"]
    print(f"{size}: " + " | ".join(parts), flush=True)
    sp_ = rec.get("spectrum")
    if sp_:
        print(f"    S D spectrum [{sp_['lam_min']:.6f}, {sp_['lam_max']:.4f}] ({sp_['time']:.1f} s), best alpha "
              f"{sp_['best_alpha']:.3f} -> rate {sp_['best_rate']:.4f}", flush=True)
    print(f"    reverse {op['reverse']:.2f}, near kink {op['near_kink']:.3f}, forward v median {op['v_fwd_median']:.3f} "
          f"p90 {op['v_fwd_p90']:.3f} max {op['v_fwd_max']:.3f} V, chord deficit median {op['deficit_median']:.3f} "
          f"p90 {op['deficit_p90']:.3f}; current share >10%: {op['current_share_deficit_over_10pct']:.2f}; "
          f"output change: linear memristor {rec['output_change_linear_memristor']:.3f}, "
          f"no diode {rec['output_change_no_diode']:.3f}", flush=True)


def load(out_dir):
    with open(Path(out_dir) / "records.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_size(records, value):
    """Mean and individual values of value(record) per array size (cells), skipping None."""
    groups = {}
    for rec in records:
        y = value(rec)
        if y is not None:
            groups.setdefault(rec["p"] * rec["q"], []).append(y)
    xs = sorted(groups)
    return np.array(xs), np.array([np.mean(groups[x]) for x in xs]), groups


def plot(out_dir):
    records = [r for r in load(out_dir) if "error" not in r]
    spec = json.loads((Path(out_dir) / "experiment.json").read_text())
    alphas = spec["alphas"]
    budget = spec["max_iterations"]
    colors = {a: SERIES_COLORS[k] for k, a in enumerate(alphas)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)

    ax = axes[0]
    for alpha in alphas:
        conv = lambda r: r["solves"][str(alpha)]["iterations"] if r["solves"][str(alpha)]["converged"] else None
        x, y, _ = by_size(records, conv)
        ax.plot(x, y, color=colors[alpha], lw=2, marker="o", ms=8, mec=SURFACE, mew=1.5, label=f"α = {alpha:g}")
        pred = lambda r: r["solves"][str(alpha)].get("predicted_iterations")
        xp, yp, _ = by_size(records, pred)
        ax.plot(xp, yp, color=colors[alpha], lw=1.5, ls=":", label=f"α = {alpha:g}: predicted from ρ")
        failed = lambda r: budget if not r["solves"][str(alpha)]["converged"] else None
        xf, yf, _ = by_size(records, failed)
        if xf.size:
            ax.plot(xf, yf, ls="none", marker="x", ms=9, mew=2, color=colors[alpha])
    ax.axhline(budget, color=MUTED, lw=1, ls="--")
    ax.text(0.02, budget * 1.15, f"iteration budget {budget} (× = not converged)", color=MUTED, fontsize=8.5,
            transform=ax.get_yaxis_transform())
    _style(ax, "Iterations to tol 1e-9", "iterations")

    ax = axes[1]
    for alpha in alphas:
        x, y, _ = by_size(records, lambda r: r["solves"][str(alpha)].get("linearized_rate"))
        ax.plot(x, y, color=colors[alpha], lw=2, marker="o", ms=8, mec=SURFACE, mew=1.5, label=f"α = {alpha:g}: ρ(J)")
        xo, yo, _ = by_size(records, lambda r: r["solves"][str(alpha)]["observed_rate"])
        ax.plot(xo, yo, ls="none", marker="D", ms=8, mfc="none", mec=colors[alpha], mew=1.5,
                label=f"α = {alpha:g}: observed")
    _style(ax, "Linearized rate ρ(J) at the solution", "rate per iteration", logy=False)
    ax.set_ylim(0, 1.02)

    ax = axes[2]
    shares = [("reverse-biased cells (diode blocks)", lambda r: r["operating_point"]["reverse"], SERIES_COLORS[3]),
              ("current through cells with chord deficit > 10%",
               lambda r: r["operating_point"]["current_share_deficit_over_10pct"], SERIES_COLORS[4]),
              ("output change, memristor made linear", lambda r: r["output_change_linear_memristor"], SERIES_COLORS[5]),
              ("output change, diodes removed", lambda r: r["output_change_no_diode"], SERIES_COLORS[6])]
    for label, value, color in shares:
        x, y, _ = by_size(records, value)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=1.5, label=label)
    _style(ax, "How much of the nonlinearity is exercised", "fraction of cells or current / relative change",
           logy=False)
    ax.set_ylim(0, None)

    c = spec["circuit"]
    kohm = lambda R: "–".join(f"{x / 1e3:g}" for x in R) if isinstance(R, list) else f"{R / 1e3:g}"
    fig.suptitle(f"Spingarn on a 1D1R crossbar (ideal diode, no leak): E ∈ [{c['E_range'][0]:g}, {c['E_range'][1]:g}] V, "
                 f"R_min {kohm(c['R_min'])} kΩ, R_max {kohm(c['R_max'])} kΩ per cell (log-uniform), γ = R_min per cell, "
                 f"{spec['seeds']} seeds per size (mean); measured on {spec['machine']}",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    path = Path(out_dir) / "diode_experiment.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def _style(ax, title, ylabel, logy=True):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("cells (p × q)", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=[(8, 8), (16, 16), (32, 32), (64, 64)])
    ap.add_argument("--seeds", type=int, default=3, help="seeds 0..SEEDS-1 per size")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.8, 1.0])
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--circuit", type=json.loads, default={}, help="JSON overrides of the 1D1R circuit for Config")
    ap.add_argument("--no-spectrum", action="store_true", help="skip the Lanczos linearized rate")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--replot", type=Path, metavar="DIR")
    args = ap.parse_args()
    if args.replot:
        print(f"wrote {plot(args.replot)}")
        return

    circuit = {**CIRCUIT, **args.circuit}
    out_dir = args.out or ROOT / "results" / datetime.now().strftime("diode-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "experiment.json").write_text(json.dumps(dict(
        started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(), sizes=args.sizes,
        seeds=args.seeds, alphas=args.alphas, max_iterations=args.max_iter, tol=TOL, circuit=circuit,
        gamma="r_lo", git_commit=git_commit()), indent=2))
    with open(out_dir / "records.jsonl", "a") as log:
        for p, q in args.sizes:
            for seed in range(args.seeds):
                rec = run_case(p, q, seed, circuit, args.alphas, args.max_iter, not args.no_spectrum)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                report(rec, args.alphas)
            plot(out_dir)
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
