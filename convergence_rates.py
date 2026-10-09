"""
Convergence of Spingarn's iteration on a small crossbar: geometric (sqrt(R_min R_max)) vs zero-bias (R_min) device
matching, against the a priori contraction bounds of algorithms.spingarns.contraction_bound.

The loop below is SpingarnSolver.solve's iteration with the stopping rule removed and instrumentation added; it is checked
against the solver itself before anything is plotted. Writes report_assets/convergence_rates.png.

    python convergence_rates.py [--size 16] [--seed 0] [--iterations 160]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Config import BranchKind, Config
from algorithms.spingarns import SpingarnSolver, contraction_bound, device_resolvent, port_resistances
from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE
from testing.test_config import solve as newton

RULES = {  # rule -> (label, color of the registered solver that uses it)
    "geometric": (r"geometric  $\gamma=\sqrt{R_{min}R_{max}}$", SERIES_COLORS[2]),
    "r_lo": (r"zero-bias  $\gamma=R_{min}$", SERIES_COLORS[0]),
}


def factorize(cfg, gamma):
    A = cfg.incidence()
    lu = spla.splu((A @ sp.diags(1 / gamma) @ A.T).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0,
                   options=dict(SymmetricMode=True))
    return A, A.T.tocsr(), lu


def iterate(cfg, gamma, alpha, iterations, a_star):
    """Run the iteration for a fixed number of passes; per pass k return ||S b^k - a^k||, ||a^k - a*|| (both in the
    Gamma^{-1} norm), the solver's relative fixed-point residual, and max |v| over devices."""
    A, AT, lu = factorize(cfg, gamma)
    d = cfg.slices[BranchKind.DEVICE]
    lin_gain = cfg.r_hi / (cfg.r_hi + gamma)
    lin_offset = gamma * cfg.emf / (cfg.r_hi + gamma)
    weight = 1 / gamma
    norm = lambda x: np.sqrt((x * x) @ weight)
    a, v = np.zeros(cfg.num_branches), np.zeros(cfg.num_branches)
    out = {key: [] for key in ("residual", "error", "solver_residual", "v_dev_max")}
    for _ in range(iterations):
        v_dev, _ = device_resolvent(a[d], gamma[d], cfg.R_min, cfg.R_max, v[d])
        v = lin_gain * a + lin_offset
        v[d] = v_dev
        b = 2 * v - a
        Sb = 2 * (AT @ lu.solve(A @ (b * weight))) - b
        out["residual"].append(norm(Sb - a))
        out["error"].append(norm(a - a_star))
        out["solver_residual"].append(norm(Sb - a) / max(norm(a), norm(Sb), 1e-300))
        out["v_dev_max"].append(np.abs(v_dev).max())
        a = (1 - alpha) * a + alpha * Sb
    return {key: np.array(val) for key, val in out.items()}


def linearized_rate(cfg, gamma, alpha, v_star):
    """Spectral radius of the iteration's Jacobian at the fixed point, (1 - alpha) I + alpha S D with
    D = diag((R_e - gamma_e) / (R_e + gamma_e)) at the solution's incremental resistances R_e: the asymptotic rate.
    Also returns max |D_e|, the operator-norm bound on that Jacobian in the scaled waves."""
    A, AT, lu = factorize(cfg, gamma)
    R = 1 / cfg.conductance(v_star)
    D = (R - gamma) / (R + gamma)
    M = AT @ lu.solve((A @ sp.diags(1 / gamma)).toarray())  # dense m x m projector
    J = (1 - alpha) * np.eye(cfg.num_branches) + alpha * (2 * M - np.eye(cfg.num_branches)) * D[None, :]
    return float(np.abs(np.linalg.eigvals(J)).max()), float(np.abs(D).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", default="report_assets/convergence_rates.png")
    args = parser.parse_args()

    cfg = Config(args.size, args.size, seed=args.seed)
    u_star, _ = newton(cfg)
    v_star = cfg.incidence().T @ u_star
    d = cfg.slices[BranchKind.DEVICE]
    print(f"{args.size}x{args.size} crossbar, seed {args.seed}, alpha {args.alpha}: {cfg.num_branches} branches, "
          f"max E {cfg.E.max():.3f} V, device |v*| in [{np.abs(v_star[d]).min():.2e}, {np.abs(v_star[d]).max():.3f}] V")

    runs = {}
    for rule in RULES:
        gamma = port_resistances(cfg, rule)
        a_star = v_star + gamma * cfg.current(v_star)
        run = iterate(cfg, gamma, args.alpha, args.iterations, a_star)
        run["bound"] = contraction_bound(cfg, gamma, args.alpha)
        run["linearized"], run["local_bound"] = linearized_rate(cfg, gamma, args.alpha, v_star)

        # The instrumented loop must be the solver's iteration: same residual history, same stopping point.
        solved = SpingarnSolver(tol=1e-9, gamma=rule, alpha=args.alpha).solve(cfg)
        k = solved.iterations
        assert np.allclose(run["solver_residual"][:k], solved.history["fixed_point_history"], rtol=1e-6, atol=1e-15)
        run["solver_iterations"] = k

        r = run["residual"] / run["residual"][0]
        e = run["error"] / run["error"][0]
        above = np.flatnonzero(r > 1e-13)  # stay clear of the rounding floor when fitting
        lo, hi = min(5, above[-1] // 2), above[-1]
        run["observed"] = (r[hi] / r[lo]) ** (1 / (hi - lo))
        print(f"\n{rule}: gamma_dev = {gamma[d][0]:.0f} ohm, solver stops (tol 1e-9) after {k} iterations")
        print(f"  a priori bound          {run['bound']:.4f}")
        print(f"  max |D_e| at solution   {run['local_bound']:.4f}")
        print(f"  linearized rate rho(SD) {run['linearized']:.4f}")
        print(f"  observed rate, k={lo}..{hi}  {run['observed']:.4f}")
        print(f"  per-step ratios r[k+1]/r[k], first 8: {np.array2string(r[1:9] / r[:8], precision=4)}")
        print(f"  per-step ratios r[k+1]/r[k], last 4 above the floor: "
              f"{np.array2string(r[hi - 3:hi + 1] / r[hi - 4:hi], precision=4)}")
        print(f"  rounding floor {r[-1]:.1e}; error and residual differ by at most "
              f"{np.max(np.abs(np.log10(e[:hi] / r[:hi]))):.2f} decades")
        print(f"  max over iterates of |v_dev| = {run['v_dev_max'].max():.3f} V (at the solution "
              f"{np.abs(v_star[d]).max():.3f} V)")
        runs[rule] = run

    plot(runs, cfg, args)
    print(f"\nwrote {args.out}")


def plot(runs, cfg, args):
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    n = args.iterations
    k = np.arange(1, n + 1)  # iteration number, as the solver counts it
    floor = 1e-17
    for rule, run in runs.items():
        label, color = RULES[rule]
        r = run["residual"] / run["residual"][0]
        few = rule == "r_lo"
        last = np.flatnonzero(r > 1e-14)[-1] + 2  # draw one point onto the rounding floor, not the flat tail
        ax.plot(k[:last], r[:last], color=color, lw=2, marker="o" if few else None, ms=5, mec=SURFACE, mew=1.5,
                label=f"{label}: measured", zorder=3)
        bound = run["bound"] ** (k - 1)
        ax.plot(k, np.where(bound > floor, bound, np.nan), color=color, lw=1.5, ls="--",
                label=f"{label}: a priori bound {run['bound']:.3f}" + r"$^{k-1}$", zorder=2)
        if few:  # for geometric the linearized rate (0.8175) sits on the bound's line, so it is only quoted
            lin = run["linearized"] ** (k - 1)
            ax.plot(k, np.where(lin > floor, lin, np.nan), color=color, lw=1.5, ls=":",
                    label=f"{label}: linearized rate {run['linearized']:.4f}" + r"$^{k-1}$", zorder=2)
        stop = run["solver_iterations"]
        ax.plot(stop, r[stop - 1], "o", ms=11, mfc="none", mec=INK_2, mew=1.2, zorder=4)
        ax.annotate(f"solver stops at tol 1e-9\n({stop} iterations)", (stop, r[stop - 1]),
                    xytext=(10, 14 if few else 10), textcoords="offset points", color=INK_2, fontsize=8.5)

    ax.axhspan(floor, 2e-15, color=GRID, alpha=0.5, lw=0, zorder=1)
    ax.text(n, 4e-16, "rounding floor", ha="right", va="center", color=MUTED, fontsize=8.5)
    g, z = runs["geometric"], runs["r_lo"]
    ax.set_title(f"Spingarn convergence on a {cfg.p}×{cfg.q} crossbar: geometric vs zero-bias device matching",
                 loc="left", color=INK, fontsize=12, pad=26)
    ax.text(0, 1.02, f"α = {args.alpha:g}, E ∈ [{cfg.E_range[0]:g}, {cfg.E_range[1]:g}] V, seed {cfg.seed}.  "
            f"Observed rates: geometric {g['observed']:.3f} (linearized {g['linearized']:.4f}), "
            f"zero-bias {z['observed']:.4f}", transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.set_xlabel("iteration k", color=INK_2)
    ax.set_ylabel(r"fixed-point residual $\|S_\Gamma b^k - a^k\|_{\Gamma^{-1}}$, relative to k = 1", color=INK_2)
    ax.set_yscale("log")
    ax.set_ylim(floor, 3)
    ax.set_xlim(0, n + 1)
    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=20))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86), frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
