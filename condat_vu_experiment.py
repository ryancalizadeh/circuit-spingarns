"""
Studies of the Condat-Vu solver (algorithms/condat_vu.py) on the memristor crossbar, against Spingarn's method and SPICE,
for report_condat_vu.md. Every solve runs in this process, so the runtimes recorded here are for orientation only; the
timed comparison goes through run_experiments.py.

    tune          iterations against the step balance rho at each size, with the observed rate and Chambolle and Pock's
                  bound for that rho
    spectrum      the spectral radius of the linearized iteration at the solution (dense, small arrays) against the
                  observed rate and the bound, for Condat-Vu and for Spingarn
    trees         the two spanning trees: ||M||, mu_R mu_G and iterations
    nonlinearity  every solver at several source ranges, with how much the memristors' tanh changes the answer
    profile       where an iteration's time goes, for Condat-Vu and Spingarn

Chambolle and Pock's linear rate (J. Math. Imaging Vis. 40 (2011) 120, Theorem 3), stated for any rho: with R strongly
monotone with constant mu_R (the smallest incremental resistance of a link) and G with mu_G (the smallest incremental
conductance of a tree branch), taking mu = 2 min(mu_R tau, mu_G sigma) satisfies the theorem's hypotheses whenever
tau sigma ||M||^2 <= 1, and a squared weighted distance to the solution shrinks at least by omega = 2 / (2 + mu) per
iteration (the distance by sqrt(omega)). With the step product theta and rho = sqrt(sigma / tau),
mu = 2 sqrt(theta) min(mu_R / rho, mu_G rho) / ||M||, largest at rho = sqrt(mu_R / mu_G). omega is also the slower of the
two sides' weakest single-branch damping, 1 / (1 + tau mu_R) and 1 / (1 + sigma mu_G); the spectrum study shows the
iteration's actual rate equals omega for rho >= sqrt(mu_R / mu_G) on the default crossbar.

    python condat_vu_experiment.py tune --sizes 8 16 32 64 128 --out results/cv-tune
    python condat_vu_experiment.py spectrum --sizes 4 8 16 24 --rhos 300 500 1000 3000 --out results/cv-spectrum-modes
    python condat_vu_experiment.py trees --sizes 1 2 3 4 6 8 --out results/cv-trees
    python condat_vu_experiment.py nonlinearity --sizes 16 64 --out results/cv-nonlinearity
    python condat_vu_experiment.py profile --sizes 64 128 256 --rho 1000 --out results/cv-profile

(report_condat_vu.md §9 lists every command its results came from.)

Writes experiment.json and records.jsonl (one record per case, appended as each finishes) to --out.
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from Config import BranchKind, Config
from algorithms.condat_vu import CondatVuSolver, Interconnect, LinesInterconnect, balanced_rho, spanning_tree
from algorithms.spingarns import SpingarnSectorSolver, SpingarnSolver, port_resistances, reflection
from diode_experiment import output_change, variant
from run_experiments import git_commit, parse_size

ROOT = Path(__file__).resolve().parent
TOL = 1e-9
RHOS = [150, 200, 300, 400, 500, 700, 1000, 1500, 3000]


def cp_bound(cfg, tree, norm_M, rho, step_product):
    """Chambolle and Pock's per-iteration contraction factor omega for this rho (module docstring), and mu_R mu_G."""
    mu_R, mu_G = cfg.r_lo[~tree].min(), 1 / cfg.r_hi[tree].max()
    mu = 2 * np.sqrt(step_product) * min(mu_R / rho, mu_G * rho) / norm_M
    return 2 / (2 + mu), float(mu_R * mu_G)


def tail_rate(history, floor=1e-13):
    """Observed per-iteration rate over the second half of a residual history, above the rounding floor."""
    r = np.asarray(history, dtype=float)
    above = np.flatnonzero(r > floor)
    if above.size < 4:
        return None
    r = r[: above[-1] + 1]
    k0 = len(r) // 2
    return float((r[-1] / r[k0]) ** (1 / (len(r) - 1 - k0)))


def run_cv(cfg, **options):
    t = time.perf_counter()
    out = CondatVuSolver(**options).solve(cfg)
    h = out.history
    return out, dict(iterations=out.iterations, converged=out.converged, runtime=time.perf_counter() - t,
                     iteration_time=h["iteration_time"], norm_M=h["norm_M"], rho=h["rho"], tau=h["tau"],
                     sigma=h["sigma"], observed_rate=tail_rate(h["kcl_history"]),
                     fixed_point_final=h["fixed_point_history"][-1], kcl_residual_rel=h["kcl_residual_rel"])


def tune(args, log):
    for p, q in args.sizes:
        for seed in range(args.seeds):
            cfg = Config(p, q, seed=seed, **args.circuit)
            tree = spanning_tree(cfg, "lines")
            for rho in ["auto"] + args.rhos:
                _, rec = run_cv(cfg, rho=rho if rho == "auto" else float(rho), max_iterations=args.max_iter)
                rec["bound"], rec["mu_R_mu_G"] = cp_bound(cfg, tree, rec["norm_M"], rec["rho"], 0.99)
                rec["bound_iterations"] = float(np.log(TOL) / np.log(rec["bound"]))
                yield dict(p=p, q=q, seed=seed, rho_option=rho, **rec)


def cv_jacobian(cfg, tau, sigma, v):
    """
    Spectrum of the Condat-Vu iteration's Jacobian at the solution v (line tree), from the dense matrix: with
    P_R = (I + tau D_R)^-1 and P_G = (I + sigma D_G)^-1 at the incremental resistances D_R of the links and the
    incremental conductances D_G of the tree branches,

        J = [[P_R, -tau P_R M^T], [sigma P_G M (2 P_R - I), P_G (I - 2 sigma tau M P_R M^T)]].

    Returns the spectral radius and two properties of the slowest mode (di, dv), measured in the iteration's metric
    ||di||^2 / tau + ||dv||^2 / sigma: link_share, the share of that mode on the links, and coupling, how strongly M acts
    on its link part relative to M's largest gain, ||M di|| / (||M|| ||di||) (0 for a link pattern M does not see, 1 for
    M's top singular vector).
    """
    net = LinesInterconnect(cfg, spanning_tree(cfg, "lines"))
    M = net.dense()
    g = cfg.conductance(v)
    P_R = 1 / (1 + tau / g[net.links])
    P_G = 1 / (1 + sigma * g[net.tree])
    top = np.hstack([np.diag(P_R), -tau * P_R[:, None] * M.T])
    bottom = np.hstack([sigma * P_G[:, None] * M * (2 * P_R - 1)[None, :],
                        P_G[:, None] * (np.eye(M.shape[0]) - 2 * sigma * tau * (M * P_R[None, :]) @ M.T)])
    lam, vec = np.linalg.eig(np.vstack([top, bottom]))
    k = int(np.argmax(np.abs(lam)))
    di, dv = vec[:M.shape[1], k], vec[M.shape[1]:, k]
    link, tree = np.vdot(di, di).real / tau, np.vdot(dv, dv).real / sigma
    coupling = np.linalg.norm(M @ di) / (np.linalg.norm(M, 2) * max(np.linalg.norm(di), 1e-300))
    return float(np.abs(lam[k])), float(link / (link + tree)), float(coupling)


def spingarn_jacobian_radius(cfg, gamma, v):
    """Spectral radius of Spingarn's linearized iteration S D at the solution (alpha = 1; dense, small arrays)."""
    A = cfg.incidence().toarray()
    W = 1 / gamma
    K = (A * W) @ A.T
    S = 2 * A.T @ np.linalg.solve(K, A * W) - np.eye(cfg.num_branches)
    D = reflection(1 / cfg.conductance(v), gamma)
    return float(np.abs(np.linalg.eigvals(S * D[None, :])).max())


def spectrum(args, log):
    for p, q in args.sizes:
        cfg = Config(p, q, seed=0)
        tree = spanning_tree(cfg, "lines")
        for rho in ["auto"] + args.rhos:
            # a tight tolerance gives a long tail to measure the rate on
            out, rec = run_cv(cfg, rho=rho if rho == "auto" else float(rho), max_iterations=args.max_iter)
            tight, _ = run_cv(cfg, rho=rho if rho == "auto" else float(rho), tol=1e-13, max_iterations=args.max_iter)
            rec["observed_rate"] = tail_rate(tight.history["kcl_history"], floor=1e-14)
            t = time.perf_counter()
            rec["linearized_rate"], rec["slow_mode_link_share"], rec["slow_mode_coupling"] = cv_jacobian(
                cfg, rec["tau"], rec["sigma"], out.v)
            rec["eig_time"] = time.perf_counter() - t
            # the slowest element's own damping per iteration, links and tree: omega is the larger
            g = cfg.conductance(out.v)
            rec["link_damping"] = float(1 / (1 + rec["tau"] / g[~tree].max()))
            rec["tree_damping"] = float(1 / (1 + rec["sigma"] * g[tree].min()))
            rec["bound"], rec["mu_R_mu_G"] = cp_bound(cfg, tree, rec["norm_M"], rec["rho"], 0.99)
            yield dict(p=p, q=q, seed=0, solver="condat_vu", rho_option=rho, **rec)
        for cls in (SpingarnSolver, SpingarnSectorSolver):
            solver = cls(tol=1e-13)
            out = solver.solve(cfg)
            gamma = port_resistances(cfg, solver.options["gamma"])
            yield dict(p=p, q=q, seed=0, solver=cls.name, iterations=cls().solve(cfg).iterations,
                       observed_rate=tail_rate(out.history["fixed_point_history"], floor=1e-14),
                       linearized_rate=spingarn_jacobian_radius(cfg, gamma, out.v),
                       bound=out.history["contraction_bound"])


def trees(args, log):
    for p, q in args.sizes:
        cfg = Config(p, q, seed=0)
        for rule in ("lines", "devices"):
            tree = spanning_tree(cfg, rule)
            _, rec = run_cv(cfg, tree=rule, max_iterations=args.max_iter)
            rec["bound"], rec["mu_R_mu_G"] = cp_bound(cfg, tree, rec["norm_M"], rec["rho"], 0.99)
            rec["bound_iterations"] = float(np.log(TOL) / np.log(rec["bound"]))
            yield dict(p=p, q=q, seed=0, tree=rule, **rec)


def operating_point(cfg, v, i):
    """How far the memristors sit from linear at the solution: their voltages, and the chord deficit 1 - (i / v) R_min,
    the fall of the chord conductance below its zero-bias value (~v^2 / 3 for small v)."""
    d = cfg.slices[BranchKind.DEVICE]
    x, cur = v[d], i[d]
    nz = np.abs(x) > 0
    deficit = 1 - cur[nz] / x[nz] * cfg.R_min_cell[nz]
    q = lambda a, s: float(np.quantile(a, s))
    return dict(v_median=q(np.abs(x), 0.5), v_p90=q(np.abs(x), 0.9), v_max=float(np.abs(x).max()),
                deficit_median=q(deficit, 0.5), deficit_p90=q(deficit, 0.9), deficit_max=float(deficit.max()))


def nonlinearity(args, log):
    from algorithms.spice import SpiceSolver
    spice = SpiceSolver()
    for lo, hi in args.e_ranges:
        for p, q in args.sizes:
            for seed in range(args.seeds):
                cfg = Config(p, q, seed=seed, **{**args.circuit, "E_range": (lo, hi)})
                rec = dict(p=p, q=q, seed=seed, E_range=[lo, hi], solvers={})
                ref = None
                for name, make in [("spingarn", SpingarnSolver), ("spingarn_sector", SpingarnSectorSolver),
                                   ("condat_vu", CondatVuSolver),
                                   ("condat_vu_rho", lambda: CondatVuSolver(rho=args.rho)), ("spice", lambda: spice)]:
                    t = time.perf_counter()
                    out = make().solve(cfg)
                    its = out.history.get("spice_iterations", out.iterations) if name == "spice" else out.iterations
                    rec["solvers"][name] = dict(iterations=int(its), converged=out.converged,
                                                runtime=time.perf_counter() - t)
                    ref = out if name == "spingarn" else ref
                rec["operating_point"] = operating_point(cfg, ref.v, ref.i)
                linear = SpingarnSolver().solve(variant(cfg, linear=True))
                rec["output_change_linear_memristor"] = output_change(cfg, ref.i, linear.i)
                rec["norm_M"] = LinesInterconnect(cfg, spanning_tree(cfg, "lines")).norm()
                yield rec


def profile(args, log):
    """
    Where an iteration's time goes, for Condat-Vu and for Spingarn (sector-matched, so that it runs as many iterations
    as asked): a fixed number of iterations with timers wrapped around the local solves, the network step and SPICE's
    KCL test; the rest of the loop is the remainder. One in-process run per solver and size, for orientation.
    """
    import algorithms.condat_vu as cv
    import algorithms.spingarns as sg
    from contextlib import ExitStack
    from unittest import mock

    clock = {}

    def timed(part, f):
        def wrapper(*a, **kw):
            t = time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                clock[part] = clock.get(part, 0.0) + time.perf_counter() - t
        return wrapper

    factorize = sg.factorize

    def timed_factorize(*a, **kw):  # time the triangular solves with the factor, not the factorization
        solve, info = factorize(*a, **kw)
        return timed("network step", solve), info

    patches = {
        "condat_vu": [(cv, "link_resolvent", timed("local solves", cv.link_resolvent)),
                      (cv, "kcl_check", timed("KCL test", cv.kcl_check))]
                     + [(cv.LinesInterconnect, m, timed("network step", getattr(cv.LinesInterconnect, m)))
                        for m in ("tree_currents", "potentials", "link_voltages")],
        "spingarn": [(sg, "cell_resolvent", timed("local solves", sg.cell_resolvent)),
                     (sg, "kcl_check", timed("KCL test", sg.kcl_check)), (sg, "factorize", timed_factorize)],
    }
    for p, q in args.sizes:
        cfg = Config(p, q, seed=0, **args.circuit)
        for name, solver in [("condat_vu", CondatVuSolver(rho=args.rho, tol=1e-30, max_iterations=args.iterations)),
                             ("spingarn", SpingarnSectorSolver(tol=1e-30, max_iterations=args.iterations))]:
            solver.solve(Config(4, 4, seed=0))  # warm up
            clock.clear()
            with ExitStack() as stack:
                for owner, attr, new in patches[name]:
                    stack.enter_context(mock.patch.object(owner, attr, new))
                out = solver.solve(cfg)
            loop = out.history["iteration_time"]
            rec = dict(p=p, q=q, seed=0, solver=name, iterations=out.iterations, loop_time=loop,
                       setup_time=out.history["setup_time"],
                       parts={part: t / out.iterations for part, t in sorted(clock.items())})
            rec["parts"]["rest of the loop"] = loop / out.iterations - sum(rec["parts"].values())
            yield rec


def report(command, rec):
    size = f"{rec['p']}x{rec['q']} seed {rec['seed']}"
    if command == "tune":
        print(f"{size} rho {rec['rho_option']!s:>5} ({rec['rho']:7.1f}): {rec['iterations']:6d} its"
              f"{'' if rec['converged'] else ' (NOT CONVERGED)'}, rate {rec['observed_rate']}, bound {rec['bound']:.5f} "
              f"(~{rec['bound_iterations']:.0f} its), ||M|| {rec['norm_M']:.2f}, {rec['iteration_time']:.2f} s", flush=True)
    elif command == "spectrum":
        extra = ""
        if rec["solver"] == "condat_vu":
            extra = (f" rho {rec['rho_option']}; slow mode: {rec['slow_mode_link_share']:.3f} on the links, coupling "
                     f"{rec['slow_mode_coupling']:.3f}; own damping links {rec['link_damping']:.6f} tree "
                     f"{rec['tree_damping']:.6f}")
        print(f"{size} {rec['solver']}: {rec['iterations']} its, linearized {rec['linearized_rate']:.6f}, "
              f"observed {rec['observed_rate']}, bound {rec['bound']:.6f}{extra}", flush=True)
    elif command == "profile":
        parts = ", ".join(f"{k} {1e3 * v:.2f}" for k, v in rec["parts"].items())
        print(f"{size} {rec['solver']}: {rec['iterations']} its, {1e3 * rec['loop_time'] / rec['iterations']:.2f} ms per "
              f"iteration under the profiler: {parts} ms; setup {rec['setup_time']:.2f} s", flush=True)
    elif command == "trees":
        print(f"{size} tree {rec['tree']}: {rec['iterations']} its{'' if rec['converged'] else ' (NOT CONVERGED)'}, "
              f"||M|| {rec['norm_M']:.3f}, mu_R mu_G {rec['mu_R_mu_G']:.2e}, rho {rec['rho']:.1f}, bound "
              f"{rec['bound']:.6f} (~{rec['bound_iterations']:.0f} its)", flush=True)
    else:
        its = ", ".join(f"{k} {v['iterations']}" for k, v in rec["solvers"].items())
        op = rec["operating_point"]
        print(f"{size} E {rec['E_range']}: {its} | |v| median {op['v_median']:.3f} p90 {op['v_p90']:.3f} max "
              f"{op['v_max']:.3f} V, deficit median {op['deficit_median']:.4f} max {op['deficit_max']:.4f}, output change"
              f" if linear {rec['output_change_linear_memristor']:.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("command", choices=["tune", "spectrum", "trees", "nonlinearity", "profile"])
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=[(8, 8), (16, 16)])
    ap.add_argument("--seeds", type=int, default=1, help="seeds 0..SEEDS-1 per size")
    ap.add_argument("--rhos", nargs="+", type=float, default=RHOS)
    ap.add_argument("--rho", type=float, default=500.0, help="nonlinearity: the fixed rho of the second Condat-Vu run")
    ap.add_argument("--e-ranges", nargs="+", type=lambda s: tuple(map(float, s.split(","))),
                    default=[(0.1, 0.5), (0.5, 1.5), (1.0, 3.0)], help="nonlinearity: source ranges as LO,HI")
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--iterations", type=int, default=200, help="profile: iterations per solver")
    ap.add_argument("--circuit", type=json.loads, default={},
                    help='tune, nonlinearity: JSON overrides for Config, e.g. \'{"R_min": 1e5, "R_max": 1e7}\'')
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out_dir = args.out or ROOT / "results" / datetime.now().strftime(f"cv-{args.command}-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "experiment.json").write_text(json.dumps(dict(
        command=args.command, started=datetime.now().isoformat(timespec="seconds"), machine=platform.node(),
        sizes=args.sizes, seeds=args.seeds, rhos=args.rhos, rho=args.rho, e_ranges=args.e_ranges, circuit=args.circuit,
        max_iterations=args.max_iter, tol=TOL, git_commit=git_commit()), indent=2))
    study = dict(tune=tune, spectrum=spectrum, trees=trees, nonlinearity=nonlinearity, profile=profile)[args.command]
    with open(out_dir / "records.jsonl", "a") as log:
        for rec in study(args, log):
            log.write(json.dumps(rec) + "\n")
            log.flush()
            report(args.command, rec)
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
