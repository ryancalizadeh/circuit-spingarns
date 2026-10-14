import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Config import Config, BranchKind
from algorithms import SOLVERS
from algorithms.scellier import CrossbarState, ScellierLinesSolver, ScellierSolver, tridiagonal_solve
from run_experiments import run_sweep
from test_config import solve as newton

DIODE = dict(device="1d1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
SELECTOR = dict(device="1s1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5), R_on=1e3, R_off=1e6, V_th=0.5,
                delta=0.0)


def test_matches_newton_reference():
    """Both block choices agree with an independent Newton solve on every cell type, and the answer satisfies the
    branch laws (to rounding: 1S1R currents come from a warm-started series solve)."""
    for circuit in ({}, DIODE, SELECTOR):
        for p, q in [(1, 1), (3, 4), (8, 8)] + ([(16, 12)] if circuit is not SELECTOR else []):
            cfg = Config(p, q, seed=p * 100 + q, **circuit)
            u_ref, _ = newton(cfg)
            v_ref = cfg.incidence().T @ u_ref
            for solver in (ScellierSolver(check_every=7), ScellierLinesSolver()):
                out = solver.solve(cfg)
                assert out.converged, (p, q, circuit, solver.name)
                assert np.abs(out.v - v_ref).max() < 1e-8 * np.abs(v_ref).max(), (p, q, circuit, solver.name)
                assert np.allclose(out.i, cfg.current(out.v), rtol=1e-9, atol=1e-15)


def test_node_update_solves_the_node_alone():
    """After a half-sweep every node of that colour satisfies its own KCL equation with its neighbours as they were: an
    exact coordinate minimization."""
    cfg = Config(5, 6, seed=3, **DIODE)
    A = cfg.incidence()
    state = CrossbarState(cfg)
    state.W[:], state.B[:] = np.random.default_rng(0).normal(scale=0.3, size=(2, 5, 6))
    state.word_colour = state.word_colour[:1]  # one half-sweep
    state.node_sweep()
    u = state.potentials()
    residual = np.abs(A @ cfg.current(A.T @ u))
    scale = abs(A) @ np.abs(cfg.current(A.T @ u))
    w = state.word_colour[0]
    updated = np.concatenate([w.ravel(), (~w).ravel()])
    assert np.all(residual[updated] <= 1e-12 * scale[updated] + 1e-15)
    assert np.any(residual[~updated] > 1e-6 * scale[~updated])  # the other colour is not yet solved


def test_line_blocks_cross_the_diode_kink():
    """At 128x128 plain Newton cycled between two states at a 1D1R cell's kink; the safeguarded line blocks solve it."""
    out = ScellierLinesSolver().solve(Config(128, 128, seed=0, **DIODE))
    assert out.converged


def test_tridiagonal_solve():
    rng = np.random.default_rng(1)
    diag, off, rhs = 3 + rng.random(9), -rng.random(8), rng.normal(size=9)
    M = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    assert np.allclose(M @ tridiagonal_solve(diag, off, rhs), rhs, rtol=0, atol=1e-13)


def test_zero_sources_give_zero_solution():
    for solver in (ScellierSolver(), ScellierLinesSolver()):
        out = solver.solve(Config(3, 3, E_range=(0, 0), seed=0, **{k: v for k, v in DIODE.items() if k != "E_range"}))
        assert out.converged and out.iterations == 1 and not out.v.any() and not out.i.any()


def test_stops_at_the_iteration_budget():
    out = ScellierSolver(max_iterations=25, check_every=10).solve(Config(4, 4, seed=0))
    assert not out.converged and out.iterations == 25 and len(out.history["kcl_history"]) == 3


def test_rejects_bad_options():
    for kwargs in [dict(blocks="rows"), dict(check_every=0), dict(check_every=2.0), dict(max_iterations=0)]:
        try:
            ScellierSolver(**kwargs)
        except ValueError:
            continue
        assert False, f"accepted {kwargs}"


def test_runs_in_the_experiment_runner():
    assert SOLVERS["scellier"] is ScellierSolver and SOLVERS["scellier_lines"] is ScellierLinesSolver
    with tempfile.TemporaryDirectory() as tmp:
        solvers = [ScellierSolver(check_every=10), ScellierLinesSolver()]
        records = run_sweep([(2, 2), (4, 4)], 1, solvers, tmp, save_pickles=False, circuit=DIODE, isolate=False)
        assert [r["status"] for r in records] == ["converged"] * 4
        spec = json.load(open(Path(tmp) / "experiment.json"))
        assert [s["blocks"] for s in spec["solvers"]] == ["nodes", "lines"]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
