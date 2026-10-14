"""
Adaptive matching on the 1D1R crossbar: the spectrum of the linearized iteration at the matching the adaptive solver
ends with, against zero-bias matching, and the rate it predicts against the rate observed after the last re-match.

At the solution the iteration is linear, J = (1 - alpha) I + alpha S D, D the branch reflection factors. Under the final
adaptive matching, conducting cells are matched near their incremental resistance (D ~ 0) and open cells at
open_resistance, where they reflect fully (D = +1), so D >= 0 and diode_experiment.linearized_spectrum applies: Lanczos on
the symmetric D^1/2 S~ D^1/2 gives the extreme eigenvalues of S D, and rho(J) = max |1 - alpha + alpha lam| over them.
The split variant relaxes branch by branch, J = I - Lam + Lam S D, which is not symmetrizable; its spectral radius comes
from Arnoldi (eigs) on J itself.
The observed rate is the fixed-point residual's per-iteration ratio from the last re-match to convergence (geometric
mean); counts and rates here do not depend on the machine.

    python adaptive_spectrum.py --sizes 16 32 64 128 --seeds 2 --out results/adaptive-spectrum

Writes results/adaptive-spectrum-<timestamp>/ (or --out): experiment.json and records.jsonl, one record per size and
seed, appended as each finishes.
"""
import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Config import BranchKind, Config
from algorithms.ordering import nested_dissection
from algorithms.spingarns import (SpingarnAdaptiveSolver, SpingarnAdaptiveSplitSolver, SpingarnSolver, blas_threads,
                                  factorize, port_resistances, reflection)
from diode_experiment import CIRCUIT, linearized_spectrum
from run_experiments import git_commit, parse_size

ROOT = Path(__file__).resolve().parent


def final_matching(cls, cfg, **options):
    """Solve cfg with the adaptive solver cls and return the output and the port resistances it ended with."""
    solver = cls(**options)
    states = []
    make = solver._matching

    def keep(c):  # hold on to the solve's re-matching state to read its last port resistances back
        states.append(make(c))
        return states[-1]

    solver._matching = keep
    out = solver.solve(cfg)
    gamma = port_resistances(cfg, solver.options["gamma"])
    d = cfg.slices[BranchKind.DEVICE]
    if states[0].last is not None:
        gamma[d] = states[0].last
    return out, gamma


def tail_rate(out):
    """Geometric-mean per-iteration ratio of the fixed-point residual from the iteration after the last re-match on."""
    r = np.asarray(out.history["fixed_point_history"])
    log = out.history["rematch_log"]
    k0 = log[-1][0] + 1 if log else 0
    n = len(r) - 1 - k0
    return float((r[-1] / r[k0]) ** (1 / n)) if n > 0 and r[k0] > 0 else None, k0


def rho(lam_min, lam_max, alpha):
    return max(abs(1 - alpha + alpha * lam_min), abs(1 - alpha + alpha * lam_max))


def extreme_eigenvalues(cfg, gamma, v, tol=1e-6, ncv=80):
    """
    As diode_experiment.linearized_spectrum, at a looser tolerance and with None for an eigenvalue ARPACK does not
    find. Under the final adaptive matching most eigenvalues of S D crowd near 0 (every conducting cell is matched) and
    lam_max sits just above them, which the 1e-10 tolerance there does not reach from 64x64 up; three digits suffice for
    the rate.
    """
    A = cfg.incidence()
    AT = A.T.tocsr()
    with blas_threads(1):
        solve, _ = factorize(A @ sp.diags(1 / gamma) @ AT, "cholmod", nested_dissection(cfg))
    with np.errstate(divide="ignore"):
        D = reflection(1 / cfg.conductance(v), gamma)
    s, g = np.sqrt(np.clip(D, 0, None)), np.sqrt(gamma)

    def matvec(x):
        z = g * (s * np.ravel(x))
        return s * ((2 * (AT @ solve(A @ (z / gamma))) - z) / g)

    op = spla.LinearOperator((cfg.num_branches,) * 2, matvec=matvec, dtype=float)
    out = []
    for which in ("SA", "LA"):
        try:
            out.append(float(spla.eigsh(op, k=1, which=which, ncv=ncv, tol=tol, maxiter=20000,
                                        return_eigenvectors=False)[0]))
        except spla.ArpackNoConvergence:
            out.append(None)
    return out[0], out[1], int(np.sum(D == 1.0))


def split_rate(cfg, gamma, v, alpha, open_alpha, open_resistance):
    """Spectral radius of J = I - Lam + Lam S D at the solution v: Lam = open_alpha on the cells matched as open,
    alpha elsewhere."""
    A = cfg.incidence()
    AT = A.T.tocsr()
    with blas_threads(1):
        solve, _ = factorize(A @ sp.diags(1 / gamma) @ AT, "cholmod", nested_dissection(cfg))
    with np.errstate(divide="ignore"):
        D = reflection(1 / cfg.conductance(v), gamma)
    d = cfg.slices[BranchKind.DEVICE]
    lam = np.full(cfg.num_branches, alpha)
    lam[d] = np.where(gamma[d] >= open_resistance, open_alpha, alpha)

    def matvec(x):
        z = D * np.ravel(x)
        return (1 - lam) * np.ravel(x) + lam * (2 * (AT @ solve(A @ (z / gamma))) - z)

    op = spla.LinearOperator((cfg.num_branches,) * 2, matvec=matvec, dtype=float)
    try:
        return float(np.abs(spla.eigs(op, k=1, which="LM", ncv=80, tol=1e-6, maxiter=20000,
                                      return_eigenvectors=False)[0]))
    except spla.ArpackNoConvergence:
        return None


def run_case(p, q, seed):
    cfg = Config(p, q, seed=seed, **CIRCUIT)
    d = cfg.slices[BranchKind.DEVICE]
    rec = dict(p=p, q=q, seed=seed)
    ref = SpingarnSolver(alpha=0.8, max_iterations=3000).solve(cfg)
    lam_min, lam_max, _ = linearized_spectrum(cfg, port_resistances(cfg, "r_lo"), ref.v)
    rec["spingarn"] = dict(iterations=ref.iterations, lam_min=lam_min, lam_max=lam_max,
                           predicted_rate=rho(lam_min, lam_max, 0.8))
    for cls in (SpingarnAdaptiveSolver, SpingarnAdaptiveSplitSolver):
        out, gamma = final_matching(cls, cfg)
        o = cls().options
        lam_min, lam_max, open_cells = extreme_eigenvalues(cfg, gamma, out.v)
        if o["open_alpha"] is None:
            predicted = None if None in (lam_min, lam_max) else rho(lam_min, lam_max, o["alpha"])
        else:
            predicted = split_rate(cfg, gamma, out.v, o["alpha"], o["open_alpha"], o["open_resistance"])
        observed, k0 = tail_rate(out)
        rec[cls.name] = dict(
            converged=out.converged, iterations=out.iterations, rematches=out.history["rematches"],
            last_rematch=k0 - 1, lam_min=lam_min, lam_max=lam_max, fully_reflecting=open_cells,
            open_matched=int(np.sum(gamma[d] >= o["open_resistance"])), reverse_at_solution=int(np.sum(out.v[d] <= 0)),
            predicted_rate=predicted, observed_rate=observed, max_dv_vs_fixed=float(np.abs(out.v - ref.v).max()))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=[(16, 16), (32, 32), (64, 64), (128, 128)])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out_dir = args.out or ROOT / "results" / datetime.now().strftime("adaptive-spectrum-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "experiment.json").write_text(json.dumps(dict(
        started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(), sizes=args.sizes,
        seeds=args.seeds, circuit=CIRCUIT, git_commit=git_commit()), indent=2))
    with open(out_dir / "records.jsonl", "a") as log:
        for p, q in args.sizes:
            for seed in range(args.seeds):
                rec = run_case(p, q, seed)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                parts = [f"{p}x{q} seed {seed}: fixed {rec['spingarn']['iterations']} its, "
                         f"lam [{rec['spingarn']['lam_min']:.5f}, {rec['spingarn']['lam_max']:.4f}]"]
                for name in ("spingarn_adaptive", "spingarn_adaptive_split"):
                    s = rec[name]
                    fmt = lambda x, spec: "-" if x is None else format(x, spec)
                    parts.append(f"{name}: {s['iterations']} its ({s['rematches']} rm, last at {s['last_rematch']}), "
                                 f"lam [{fmt(s['lam_min'], '.5f')}, {fmt(s['lam_max'], '.4f')}], rate pred "
                                 f"{fmt(s['predicted_rate'], '.3f')} obs {fmt(s['observed_rate'], '.3f')}, "
                                 f"open {s['open_matched']} / reverse {s['reverse_at_solution']}")
                print("\n    ".join(parts), flush=True)
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
