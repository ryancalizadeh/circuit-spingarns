import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Config import Config, BranchKind
from algorithms import SOLVERS
from algorithms.condat_vu import (CondatVuSolver, Interconnect, LinesInterconnect, balanced_rho, lines_norm,
                                  link_resolvent, spanning_tree)
from devices import memristor_current
from run_experiments import run_sweep
from test_config import solve as newton

SHAPES = [(1, 1), (1, 5), (4, 1), (3, 4), (7, 2), (13, 7)]


def test_trees_span_the_circuit():
    """Both trees have one branch per node and leave the right kinds as links; a set that is not a spanning tree is
    refused."""
    for p, q in SHAPES:
        cfg = Config(p, q, seed=0)
        for rule, links in [("lines", {BranchKind.DEVICE}), ("devices", {BranchKind.BIT_WIRE, BranchKind.LOAD})]:
            tree = spanning_tree(cfg, rule)
            assert tree.sum() == cfg.num_nodes
            assert set(cfg.kind[~tree]) <= links
            Interconnect(cfg, tree)  # A_T is nonsingular
    cfg = Config(3, 4, seed=0)
    not_a_tree = spanning_tree(cfg, "lines")
    not_a_tree[cfg.slices[BranchKind.SOURCE]] = False  # word lines cut off from ground
    not_a_tree[cfg.slices[BranchKind.DEVICE]] = True
    not_a_tree[np.flatnonzero(not_a_tree)[cfg.num_nodes:]] = False
    try:
        Interconnect(cfg, not_a_tree)
    except RuntimeError:  # SuperLU: singular matrix
        pass
    else:
        assert False, "accepted a set of branches that does not span the circuit"


def test_interconnect_is_kcl_and_kvl():
    """M i_L are the tree currents that make every node's currents sum to zero; the potentials of v_T reproduce v_T on
    the tree; M^T is M's adjoint; and the line sweeps agree with the solves with A_T."""
    rng = np.random.default_rng(0)
    for p, q in SHAPES:
        cfg = Config(p, q, seed=1)
        A = cfg.incidence()
        for rule in ("lines", "devices"):
            tree = spanning_tree(cfg, rule)
            nets = [Interconnect(cfg, tree, A)] + ([LinesInterconnect(cfg, tree, A)] if rule == "lines" else [])
            for net in nets:
                iL, vT = rng.normal(size=net.links.size), rng.normal(size=net.tree.size)
                iT, u = net.tree_currents(iL), net.potentials(vT)
                assert np.abs(A[:, net.tree] @ iT + A[:, net.links] @ iL).max() < 1e-13
                assert np.abs(A[:, net.tree].T @ u - vT).max() < 1e-13
                assert np.allclose(net.link_voltages(u), A[:, net.links].T @ u, rtol=0, atol=1e-13)
                assert abs(iT @ vT - iL @ net.transpose(vT)) < 1e-12 * np.linalg.norm(iT) * np.linalg.norm(vT)
                assert np.allclose(net.dense() @ iL, iT, rtol=0, atol=1e-13)


def test_norm_of_M():
    """||M|| for the line tree in closed form agrees with the dense matrix's and with Lanczos, and so does its smallest
    singular value; ||M|| grows in proportion to the side of a square array, about 0.90 (n + 1/2)."""
    for p, q in SHAPES + [(9, 9), (20, 12)]:
        cfg = Config(p, q, seed=0)
        tree = spanning_tree(cfg, "lines")
        general, lines = Interconnect(cfg, tree), LinesInterconnect(cfg, tree)
        singular = np.linalg.svd(general.dense(), compute_uv=False)
        assert np.isclose(lines.norm(), singular[0], rtol=1e-12), (p, q)
        assert np.isclose(lines_norm(p, q, smallest=True), singular[-1], rtol=1e-10), (p, q)
        assert np.isclose(general.norm(), singular[0], rtol=1e-9), (p, q)
    for n in (16, 64, 256):  # the approximation's error is ~(pi / (4n + 2))^2 / 6
        cfg = Config(n, n, seed=0)
        norm = LinesInterconnect(cfg, spanning_tree(cfg, "lines")).norm()
        assert abs(norm / (2 * np.sqrt(2) / np.pi * (n + 0.5)) - 1) < 1e-3, (n, norm)


def test_link_resolvent_solves_its_equation():
    """i + tau g_m^-1(i) = x for a memristor link: w = g_m^-1(i) is the link's voltage, so g_m(w) = i and i + tau w = x,
    from far below to far above saturation, for steps from tiny to huge and warm starts on either side."""
    R_min, R_max = 1e3, 100e3
    x = np.concatenate([np.linspace(-0.05, 0.05, 2001), [0.0, 1e-300, -1e-15]])
    for tau in (1e-7, 1e-5, 1e-3, 1e-1):
        for w0 in (np.zeros_like(x), -50 * np.sign(x), np.full_like(x, 3.0)):
            i, w, steps = link_resolvent(x, tau, R_min, R_max, w0)
            # i = x - tau w cancels when tau w ~ x, so g_m(w) = i holds to rounding relative to x, not to i
            assert np.all(np.abs(memristor_current(w, R_min, R_max) - i) <= 1e-13 * np.abs(x))
            assert np.all(np.abs(i + tau * w - x) <= 4 * np.finfo(float).eps * (np.abs(i) + tau * np.abs(w) + np.abs(x)))
            assert np.all(np.sign(i) == np.sign(x)) and steps <= 12


def test_matches_newton_reference():
    """The answer agrees with an independent Newton solve, with per-cell resistances too, for several step balances and
    both trees, and satisfies KVL exactly and the branch laws by construction."""
    cases = [(p, q, {}) for p, q in [(1, 1), (4, 1), (1, 5), (3, 4), (16, 32)]]
    cases += [(5, 6, dict(R_min=(1e3, 1e4), R_max=(5e4, 2e5), E_range=(-1, 1.5)))]
    for p, q, circuit in cases:
        cfg = Config(p, q, seed=p * 100 + q, **circuit)
        u_ref, _ = newton(cfg)
        v_ref = cfg.incidence().T @ u_ref
        settings = [dict(rho=rho) for rho in ("auto", 300.0, 3e3)] + [dict(step_product=0.5)]
        if p * q <= 12:
            settings.append(dict(tree="devices", max_iterations=200_000))
        for kwargs in settings:
            out = CondatVuSolver(**kwargs).solve(cfg)
            assert out.converged, (p, q, kwargs)
            assert np.abs(out.v - v_ref).max() < 1e-8 * np.abs(v_ref).max(), (p, q, kwargs)
            assert np.array_equal(out.i, cfg.current(out.v))


def test_step_sizes():
    """tau sigma ||M||^2 is the step product and sqrt(sigma / tau) the balance rho; "auto" balances the smallest link
    resistance against the largest tree resistance, sqrt(R_min R_load) on the default crossbar with the line tree."""
    cfg = Config(6, 9, seed=0)
    h = CondatVuSolver(max_iterations=1).solve(cfg).history
    assert np.isclose(h["tau"] * h["sigma"] * h["norm_M"] ** 2, 0.99, rtol=1e-12)
    assert np.isclose(h["rho"], np.sqrt(cfg.R_min * cfg.R_load)) and np.isclose(np.sqrt(h["sigma"] / h["tau"]), h["rho"])
    h = CondatVuSolver(rho=250.0, step_product=0.5, max_iterations=1).solve(cfg).history
    assert np.isclose(h["tau"] * h["sigma"] * h["norm_M"] ** 2, 0.5) and np.isclose(np.sqrt(h["sigma"] / h["tau"]), 250)
    tree = spanning_tree(cfg, "devices")
    assert np.isclose(balanced_rho(cfg, tree), np.sqrt(min(cfg.R_wire, cfg.R_load) * cfg.R_max))


def test_iterations_grow_with_the_norm_of_M():
    """Unlike Spingarn's, the iteration count is not size-independent: it grows with ||M||, here roughly in proportion."""
    outs = [CondatVuSolver().solve(Config(n, n, seed=0)) for n in (8, 16, 32)]
    its = np.array([o.iterations for o in outs])
    norms = np.array([o.history["norm_M"] for o in outs])
    assert np.all(np.diff(its) > 0)
    assert np.all(np.abs(np.log((its[1:] / its[:-1]) / (norms[1:] / norms[:-1]))) < np.log(1.3)), (its, norms)


def test_zero_sources_give_zero_solution():
    out = CondatVuSolver().solve(Config(3, 3, E_range=(0, 0), seed=0))
    assert out.converged and out.iterations == 1 and not out.v.any() and not out.i.any()


def test_stops_at_the_iteration_budget():
    cfg = Config(4, 4, seed=0)
    out = CondatVuSolver(max_iterations=3).solve(cfg)
    assert not out.converged and out.iterations == 3
    assert len(out.history["kcl_history"]) == len(out.history["fixed_point_history"]) == 3
    assert out.v.shape == out.i.shape == (cfg.num_branches,)


def test_rejects_bad_options():
    for kwargs in [dict(rho=-1.0), dict(rho="geometric"), dict(rho=True), dict(step_product=1.0),
                   dict(step_product=0), dict(tree="bfs"), dict(max_iterations=0)]:
        try:
            CondatVuSolver(**kwargs)
        except ValueError:
            continue
        assert False, f"accepted {kwargs}"
    try:
        CondatVuSolver().solve(Config(2, 2, seed=0, device="1s1r"))
    except ValueError:
        pass
    else:
        assert False, "solved a 1s1r crossbar"
    try:  # 1D1R cells in the tree: open cells conduct nothing, so there is no automatic balance
        CondatVuSolver(tree="devices").solve(Config(2, 2, seed=0, device="1d1r"))
    except ValueError:
        pass
    else:
        assert False, "balanced the steps against open cells in the tree"


def test_1d1r_matches_newton_reference():
    """On 1D1R crossbars with signed inputs the iteration agrees with Newton, for both trees; reverse-biased cells carry
    exactly zero current."""
    for p, q in [(1, 1), (3, 4), (8, 8), (16, 12)]:
        cfg = Config(p, q, seed=p * 100 + q, device="1d1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
        u_ref, _ = newton(cfg)
        v_ref = cfg.incidence().T @ u_ref
        d = cfg.slices[BranchKind.DEVICE]
        settings = [dict(), dict(rho=300.0)] + ([dict(tree="devices", rho=100.0, max_iterations=200_000)]
                                                if p * q <= 12 else [])
        for kwargs in settings:
            out = CondatVuSolver(**kwargs).solve(cfg)
            assert out.converged, (p, q, kwargs)
            assert np.abs(out.v - v_ref).max() < 1e-8 * np.abs(v_ref).max(), (p, q, kwargs)
            assert np.all(out.i[d][out.v[d] <= 0] == 0)


def test_runs_in_the_experiment_runner():
    """Registered as condat_vu, with its options recorded, and runnable beside the other solvers."""
    assert SOLVERS["condat_vu"] is CondatVuSolver
    with tempfile.TemporaryDirectory() as tmp:
        solvers = [SOLVERS["spingarn"](), CondatVuSolver(rho=400.0)]
        records = run_sweep([(2, 2), (4, 4)], 2, solvers, tmp, save_pickles=False)
        assert [r["status"] for r in records] == ["converged"] * 8
        assert [r["algorithm"] for r in records] == ["spingarn", "condat_vu"] * 4
        assert all(len(r["history"]["kcl_history"]) == r["iterations"] for r in records)
        spec = json.load(open(Path(tmp) / "experiment.json"))
        assert spec["solvers"][1] == dict(name="condat_vu", tol=1e-9, max_iterations=100_000, rho=400.0,
                                          step_product=0.99, tree="lines", abstol=1e-12)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
