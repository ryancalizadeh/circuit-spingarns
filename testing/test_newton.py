import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Config import BranchKind, Config
from algorithms import SOLVERS
from algorithms.newton import NewtonSafeguardedSolver, NewtonSolver, NodalMatrix
from run_experiments import run_sweep
from test_config import solve as reference

CELLS = dict(E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
SELECTOR = dict(device="1s1r", R_on=1e3, R_off=1e6, V_th=0.5, **CELLS)
SHOCKLEY = dict(device="1d1r_shockley", **CELLS)


def test_matches_reference_on_every_cell():
    """Both variants agree with the independent reference Newton (SuperLU, no safeguard) on memristor, 1D1R, 1S1R (kink
    and rounded knee) and lumped Shockley 1D1R crossbars, and their answers pass the KCL test."""
    circuits = [dict(R_min=(1e3, 1e4), R_max=(5e4, 2e5)), dict(device="1d1r", **CELLS), dict(SELECTOR, delta=0.0),
                dict(SELECTOR, delta=0.05), dict(SHOCKLEY, I_s=1e-14), dict(SHOCKLEY, I_s=1e-6, n=1.5)]
    for circuit in circuits:
        for p, q in [(1, 1), (3, 4), (16, 32)]:
            cfg = Config(p, q, seed=p * 100 + q, **circuit)
            v_ref = cfg.incidence().T @ reference(cfg)[0]
            for cls in (NewtonSolver, NewtonSafeguardedSolver):
                out = cls().solve(cfg)
                key = (p, q, cfg.device, cls.name)
                assert out.converged and out.history["failure"] is None, key
                assert np.abs(out.v - v_ref).max() < 1e-9 * np.abs(v_ref).max(), key
                assert out.history["kcl_residual_rel"] <= 1e-9 or out.history["kcl_residual"] <= 1e-12, key
                assert len(out.history["kcl_history"]) == out.iterations + 1
                assert out.history["cholmod_ordering"] == "given" and out.history["lnz"] > 0


def test_lumped_and_internal_node_agree():
    """The Shockley 1D1R crossbar solved as lumped cells and with the diodes on branches of their own gives the same
    word and bit potentials, for silicon- to selector-like saturation currents."""
    for I_s in (1e-14, 1e-9, 1e-6):
        for p, q in [(1, 1), (3, 4), (16, 32)]:
            lumped = Config(p, q, seed=7, I_s=I_s, **SHOCKLEY)
            internal = Config(p, q, seed=7, I_s=I_s, internal_node=True, **SHOCKLEY)
            a, b = NewtonSolver().solve(lumped), NewtonSafeguardedSolver().solve(internal)
            assert a.converged and b.converged, (I_s, p, q)
            u_a, u_b = lumped.line_potentials(a.v), internal.line_potentials(b.v)
            assert np.abs(u_a - u_b).max() < 1e-9 * np.abs(u_a).max(), (I_s, p, q)
            d, e = internal.slices[BranchKind.DEVICE], internal.slices[BranchKind.DIODE]
            assert np.allclose(b.v[d] + b.v[e], a.v[lumped.slices[BranchKind.DEVICE]], rtol=1e-8, atol=1e-11)


def test_jacobian_keeps_its_pattern():
    """NodalMatrix forms A diag(g) A^T on a fixed pattern, storing zeros where conductances vanish, so CHOLMOD's
    analysis of the first step serves every later one."""
    for circuit in (dict(), dict(SHOCKLEY, internal_node=True)):
        cfg = Config(6, 5, seed=1, **circuit)
        A = cfg.incidence()
        build = NodalMatrix(cfg, A)
        pattern = None
        for g in (np.ones(cfg.num_branches), np.where(np.arange(cfg.num_branches) % 3, 1.0, 0.0),
                  np.random.default_rng(0).uniform(size=cfg.num_branches)):
            J = build(g)
            assert abs(J - A @ sp.diags(g) @ A.T).max() < 1e-15
            if pattern is not None:
                assert np.array_equal(J.indptr, pattern.indptr) and np.array_equal(J.indices, pattern.indices)
            pattern = J


def test_safeguard_tames_the_exponential_overshoot():
    """With the Shockley diode on its own branch, plain Newton from zero overshoots the exponential and crawls back by
    about n V_T per step, or fails outright when the overshoot drives the Jacobian beyond what Cholesky can factorize
    (recorded as a failure, not raised). Step halving cuts that to a few iterations. On the lumped cell, whose slope the
    memristor caps, plain Newton is fast and the safeguard never halves."""
    plain, safe = {}, {}
    for I_s in (1e-14, 1e-9, 1e-6):
        cfg = Config(8, 8, seed=0, I_s=I_s, internal_node=True, **SHOCKLEY)
        plain[I_s], safe[I_s] = NewtonSolver().solve(cfg), NewtonSafeguardedSolver().solve(cfg)
        assert safe[I_s].converged and safe[I_s].history["halvings"] > 0 and safe[I_s].iterations <= 15, I_s
        assert plain[I_s].history["halvings"] == 0
        # which of the two happens depends on rounding (the first step drives diode conductances to ~1e14 S beside
        # 1e-4 S memristors, past what the elimination of the internal node survives), so only either is asserted
        if plain[I_s].converged:
            assert plain[I_s].iterations > 3 * safe[I_s].iterations, I_s
            assert 0.5 < np.median(plain[I_s].history["step_max_history"][1:]) / 0.02585 < 3, I_s  # ~n V_T a step
        else:
            assert plain[I_s].history["failure"].startswith("factorization failed"), I_s
        lumped = Config(8, 8, seed=0, I_s=I_s, **SHOCKLEY)
        out = NewtonSafeguardedSolver().solve(lumped)
        assert out.converged and out.iterations <= 8 and out.history["halvings"] == 0, I_s
    assert any(out.converged for out in plain.values())


def test_refuses_ideal_diodes_on_their_own_branch_and_bad_options():
    try:
        NewtonSolver().solve(Config(3, 3, seed=0, device="1d1r", internal_node=True))
        assert False, "solved ideal diodes on their own branches"
    except ValueError:
        pass
    for kwargs in [dict(safeguard=1), dict(max_halvings=0), dict(ordering="metis"), dict(blas_threads=0),
                   dict(max_iterations=0)]:
        try:
            NewtonSolver(**kwargs)
        except ValueError:
            continue
        assert False, f"accepted {kwargs}"


def test_registered_and_runs_in_the_runner():
    assert SOLVERS["newton"] is NewtonSolver and SOLVERS["newton_safeguarded"] is NewtonSafeguardedSolver
    assert NewtonSolver().options["safeguard"] is False and NewtonSafeguardedSolver().options["safeguard"] is True
    with tempfile.TemporaryDirectory() as tmp:
        records = run_sweep([(4, 4)], 2, [NewtonSolver(), NewtonSafeguardedSolver()], tmp,
                            circuit=dict(SHOCKLEY, I_s=1e-9), save_pickles=False)
        assert [r["status"] for r in records] == ["converged"] * 4
        assert all("halvings" in r["history"] for r in records)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
