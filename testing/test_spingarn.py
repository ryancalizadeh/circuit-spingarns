import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Config import Config, BranchKind
from algorithms import SOLVERS
from algorithms.ordering import nested_dissection
from algorithms.spingarns import (SpingarnCholmodSolver, SpingarnSectorSolver, SpingarnSLUSolver, SpingarnSolver,
                                  blas_threads, device_resolvent, factorize, port_resistances)
from run_experiments import run_sweep
from test_config import solve as newton

BACKENDS = [(f, o) for f in ("cholmod", "slu") for o in ("nested_dissection", "builtin")]


def test_matches_newton_reference():
    """Spingarn's answer agrees with an independent Newton solve for every port-resistance rule and relaxation, and on
    every factorization and ordering, and satisfies the branch laws to rounding error."""
    for p, q in [(1, 1), (4, 1), (1, 5), (3, 4), (16, 32)]:
        cfg = Config(p, q, seed=p * 100 + q)
        u_ref, _ = newton(cfg)
        v_ref = cfg.incidence().T @ u_ref
        settings = [dict(gamma=gamma, alpha=alpha) for gamma in ["geometric", "r_lo", 3e3] for alpha in [1.0, 0.5]]
        settings += [dict(factorization=f, ordering=o) for f, o in BACKENDS]
        for kwargs in settings:
            out = SpingarnSolver(**kwargs).solve(cfg)
            assert out.converged, (p, q, kwargs)
            assert np.abs(out.v - v_ref).max() < 1e-8 * np.abs(v_ref).max(), (p, q, kwargs)
            assert np.abs(out.i - cfg.current(out.v)).max() < 1e-14 * np.abs(out.i).max()
            assert out.history["kvl_residual_rel"] <= 1e-9


def test_1d1r_matches_newton_reference():
    """On 1D1R crossbars with signed inputs and per-cell resistances, the averaged iteration (alpha < 1, which is what
    guarantees convergence without a leak) agrees with Newton, for every port-resistance rule; reverse-biased cells
    carry exactly zero current. The a priori bound is 1: open cells reflect fully."""
    for p, q in [(1, 1), (3, 4), (8, 8), (16, 32)]:
        cfg = Config(p, q, seed=p * 100 + q, device="1d1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
        u_ref, _ = newton(cfg)
        v_ref = cfg.incidence().T @ u_ref
        d = cfg.slices[BranchKind.DEVICE]
        for kwargs in [dict(gamma=gamma, alpha=alpha) for gamma in ["r_lo", "geometric", 3e3] for alpha in [0.5, 0.8]]:
            out = SpingarnSolver(max_iterations=5000, **kwargs).solve(cfg)
            assert out.converged, (p, q, kwargs)
            assert np.abs(out.v - v_ref).max() < 1e-7 * np.abs(v_ref).max(), (p, q, kwargs)
            assert np.all(out.i[d][out.v[d] <= 0] == 0)
            assert out.history["contraction_bound"] == 1.0


SELECTOR = dict(device="1s1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5), R_on=1e3, R_off=1e6, V_th=0.5)


def test_1s1r_matches_newton_reference():
    """On 1S1R crossbars, with a rounded knee and with the kink, Spingarn agrees with Newton for both matchings. The
    cell's sector is bounded, so alpha = 1 contracts; with geometric matching the a priori bound is the worst cell's
    (sqrt K - 1) / (sqrt K + 1), K = (R_off + R_max) / (R_on + R_min), whatever delta."""
    for delta in (0.05, 0.0):
        for p, q in [(1, 1), (3, 4), (8, 8)]:
            cfg = Config(p, q, seed=p * 100 + q, delta=delta, **SELECTOR)
            u_ref, _ = newton(cfg)
            v_ref = cfg.incidence().T @ u_ref
            d = cfg.slices[BranchKind.DEVICE]
            K = (cfg.R_off + cfg.R_max_cell) / (cfg.R_on + cfg.R_min_cell)
            for kwargs in [dict(gamma="geometric"), dict(gamma="geometric", alpha=0.5), dict(gamma="r_lo")]:
                out = SpingarnSolver(**kwargs).solve(cfg)
                assert out.converged, (p, q, delta, kwargs)
                assert np.abs(out.v - v_ref).max() < 1e-8 * np.abs(v_ref).max(), (p, q, delta, kwargs)
                assert np.allclose(out.i[d], cfg.current(out.v)[d], rtol=1e-9, atol=1e-15)
            bound = SpingarnSolver(gamma="geometric", max_iterations=1).solve(cfg).history["contraction_bound"]
            assert np.isclose(bound, np.max((np.sqrt(K) - 1) / (np.sqrt(K) + 1)), rtol=1e-12)


def test_series_resolvent():
    """devices.series_solve solves v_s + v_m + gamma i = a with i = g_s(v_s) = g_m(v_m), from deep reverse bias through
    the selector's knee to deep forward bias, for knees from soft to a kink and port resistances from tiny to huge,
    cold or warm-started from anywhere."""
    from devices import memristor_current, selector_current, series_solve
    rng = np.random.default_rng(2)
    n = 4000
    a = np.concatenate([rng.normal(scale=3, size=n - 6), [0.0, 0.5, -0.5, 0.5 + 1e-12, 1e-300, 50.0]])
    R_min, R_max = np.exp(rng.uniform(np.log(1e3), np.log(1e4), n)), np.exp(rng.uniform(np.log(5e4), np.log(2e5), n))
    for delta in (0.1, 1e-3, 0.0):
        for gamma in (1.0, 3e3, 1e5, 1e8):
            args = (gamma, R_min, R_max, 1e3, 1e6, 0.5, delta)
            for x0 in (None, (rng.normal(size=n), rng.normal(size=n))):
                v, i, v_s, v_m, steps = series_solve(a, *args, x0=x0)
                scale = np.abs(v_s) + np.abs(v_m) + gamma * np.abs(i) + np.abs(a)
                assert np.all(np.abs(v + gamma * i - a) <= 1e-11 * scale), (delta, gamma)
                assert np.allclose(selector_current(v_s, 1e3, 1e6, 0.5, delta), i, rtol=1e-11, atol=1e-18)
                assert np.allclose(memristor_current(v_m, R_min, R_max), i, rtol=1e-11, atol=1e-18)
                assert np.all(np.sign(v) == np.sign(a)) and np.array_equal(v, v_s + v_m) and steps <= 100


def test_factorize_solves_in_any_ordering():
    """Both factorizations solve the nodal system in the library's ordering, in the nested-dissection ordering and in an
    arbitrary one, so the permutation bookkeeping is right whatever the ordering."""
    cfg = Config(5, 7, seed=3)
    A = cfg.incidence()
    K = A @ sp.diags(1 / port_resistances(cfg, "r_lo")) @ A.T
    rhs = np.random.default_rng(0).standard_normal(cfg.num_nodes)
    for factorization in ("cholmod", "slu"):
        for perm in (None, nested_dissection(cfg), np.random.default_rng(1).permutation(cfg.num_nodes)):
            solve, info = factorize(K, factorization, perm)
            x = solve(rhs)
            assert np.abs(K @ x - rhs).max() < 1e-12 * np.abs(rhs).max(), (factorization, perm)
            assert np.array_equal(solve(rhs), x)  # solving again gives the same answer: no state left behind
            assert info["factor_nnz"] > 0
            if perm is not None:
                assert info["ordering_used"] == "nested_dissection"


def test_backends_agree_and_report_what_they_did():
    """Every factorization and ordering takes the same iterations to the same answer. CHOLMOD, left to itself, reports
    that it chose AMD; given the nested-dissection ordering, that it used it as given, with less fill than AMD's."""
    cfg = Config(32, 32, seed=0)
    outs = {(f, o): SpingarnSolver(factorization=f, ordering=o).solve(cfg) for f, o in BACKENDS}
    ref = outs["cholmod", "nested_dissection"]
    for (f, o), out in outs.items():
        h = out.history
        assert out.converged and out.iterations == ref.iterations, (f, o)
        assert np.abs(out.v - ref.v).max() < 1e-10 * np.abs(ref.v).max(), (f, o)
        assert h["factor_nnz"] > 0 and h["factor_time"] > 0 and h["setup_time"] >= h["factor_time"]
        assert h["ordering_used"] == {"nested_dissection": "nested_dissection", "builtin": {"slu": "mmd",
                                                                                         "cholmod": "amd"}[f]}[o]
        assert (h["ordering_time"] > 0) == (o == "nested_dissection")
        assert h["blas_threads"] == (1 if f == "cholmod" else None)
        if f == "cholmod":
            assert h["cholmod_ordering"] == ("given" if o == "nested_dissection" else "amd")
            assert 0 < h["lnz"] <= h["factor_nnz"]
    big = Config(128, 128, seed=0)
    nd, amd = (SpingarnSolver(ordering=o, max_iterations=1).solve(big).history["lnz"]
               for o in ("nested_dissection", "builtin"))
    assert nd < 0.9 * amd, (nd, amd)


def test_blas_threads_are_set_and_restored():
    with blas_threads(None) as before:
        pass
    if before is None:  # cvxopt's OpenBLAS not reachable on this platform: nothing to control
        return
    with blas_threads(2) as inside:
        assert inside == 2
    with blas_threads(None) as after:
        assert after == before
    assert SpingarnSolver(blas_threads=2).solve(Config(3, 3, seed=0)).history["blas_threads"] == 2


def test_device_resolvent_solves_its_equation():
    """v + gamma g(v) = a to rounding error, from deep negative saturation through zero bias to positive saturation,
    with warm starts on either side of the root and of the wrong sign. The left side has slope >= 1, so a small
    residual means a small error."""
    R_min, R_max = 1e3, 100e3
    a = np.concatenate([np.linspace(-200, 200, 4001), [0.0, 1e-300, -1e-12, 1e-6]])
    for gamma in [5.0, 1e3, 1e4, 1e6]:
        c, d = 1 + gamma / R_max, gamma * (1 / R_min - 1 / R_max)
        for v0 in [np.zeros_like(a), -3 * a, np.full_like(a, 50.0), a]:
            v, steps = device_resolvent(a, gamma, R_min, R_max, v0)
            residual = np.abs(c * v + d * np.tanh(v) - a)
            assert np.all(residual <= 2 * np.finfo(float).eps * (c * np.abs(v) + d * np.abs(np.tanh(v)) + np.abs(a)))
            assert np.all(np.sign(v) == np.sign(a)) and steps <= 10


def test_device_resolvent_with_diode():
    """For a 1D1R cell, v + gamma g_m(max(v, 0)) = a: v = a when a <= 0 (the cell is open), the memristor's root when
    a > 0; per-device parameters broadcast."""
    rng = np.random.default_rng(1)
    n = 5000
    a = np.concatenate([rng.normal(scale=20, size=n - 4), [0.0, -1e-300, 1e-300, 1e-9]])
    R_min, R_max = np.exp(rng.uniform(np.log(1e3), np.log(1e4), n)), np.exp(rng.uniform(np.log(5e4), np.log(2e5), n))
    gamma = R_min * rng.uniform(0.1, 10, n)
    v, _ = device_resolvent(a, gamma, R_min, R_max, rng.normal(size=n), diode=True)
    x = np.maximum(v, 0)
    i = x / R_max + (1 / R_min - 1 / R_max) * np.tanh(x)
    assert np.all(v[a <= 0] == a[a <= 0]) and np.all(v[a > 0] > 0)
    assert np.all(np.abs(v + gamma * i - a) <= 4 * np.finfo(float).eps * (np.abs(v) + gamma * i + np.abs(a)))
    plain, _ = device_resolvent(a, gamma, R_min, R_max, np.zeros(n))
    assert np.allclose(v[a > 0], plain[a > 0], rtol=1e-14, atol=0)  # same root from another warm start


def test_port_resistances():
    """Linear branches are matched exactly by every rule; only the devices' port resistance depends on it, cell by cell
    when the cells have their own resistances."""
    cfg = Config(3, 4, seed=0)
    d = cfg.slices[BranchKind.DEVICE]
    linear = cfg.kind != BranchKind.DEVICE
    for rule, device in [("geometric", 1e4), ("r_lo", 1e3), (2.5e3, 2.5e3)]:
        gamma = port_resistances(cfg, rule)
        assert np.all(gamma[linear] == cfg.r_hi[linear]) and np.allclose(gamma[d], device)
    for device in ("memristor", "1d1r"):
        cfg = Config(3, 4, seed=0, R_min=(1e3, 1e4), R_max=(5e4, 2e5), device=device)
        assert np.array_equal(port_resistances(cfg, "r_lo")[d], cfg.R_min_cell)
        assert np.allclose(port_resistances(cfg, "geometric")[d], np.sqrt(cfg.R_min_cell * cfg.R_max_cell))
        assert np.all(port_resistances(cfg, "geometric")[linear] == cfg.r_hi[linear])
    cfg = Config(3, 4, seed=0, **SELECTOR)  # a 1S1R cell is matched over its own sector
    lo, hi = 1e3 + cfg.R_min_cell, 1e6 + cfg.R_max_cell
    assert np.allclose(port_resistances(cfg, "r_lo")[d], lo)
    assert np.allclose(port_resistances(cfg, "geometric")[d], np.sqrt(lo * hi))


def test_matched_linear_circuit_solves_in_one_update():
    """With linear devices (R_min = R_max) every branch is matched and absorbs its wave, so the reflected waves do not
    depend on the incident ones and the first update lands on the solution; the second pass confirms it."""
    cfg = Config(5, 7, R_min=2e3, R_max=2e3, seed=1)
    out = SpingarnSolver(tol=1e-12).solve(cfg)
    u_ref, _ = newton(cfg)
    v_ref = cfg.incidence().T @ u_ref
    assert out.converged and out.iterations == 2 and out.history["contraction_bound"] == 0
    assert np.abs(out.v - v_ref).max() < 1e-12 * np.abs(v_ref).max()


def test_iteration_count_does_not_grow_with_size():
    """The contraction bound depends only on each branch's own sector, not on the size of the array."""
    for gamma in ["geometric", "r_lo"]:
        counts = [SpingarnSolver(gamma=gamma).solve(Config(p, p, seed=0)).iterations for p in (8, 32, 128)]
        assert max(counts) - min(counts) <= 5, (gamma, counts)


def test_zero_sources_give_zero_solution():
    out = SpingarnSolver().solve(Config(3, 3, E_range=(0, 0), seed=0))
    assert out.converged and out.iterations == 1 and not out.v.any() and not out.i.any()


def test_stops_at_the_iteration_budget():
    cfg = Config(4, 4, seed=0)
    out = SpingarnSolver(max_iterations=3).solve(cfg)
    assert not out.converged and out.iterations == 3
    assert len(out.history["kcl_history"]) == len(out.history["kvl_history"]) == 3
    assert out.v.shape == out.i.shape == (cfg.num_branches,)


def test_rejects_bad_options():
    for kwargs in [dict(gamma="median"), dict(gamma=-1.0), dict(gamma=[1e3]), dict(alpha=0), dict(alpha=1.5),
                   dict(max_iterations=0), dict(factorization="lu"), dict(ordering="metis"), dict(blas_threads=0),
                   dict(blas_threads=1.5), dict(blas_threads=True)]:
        try:
            SpingarnSolver(**kwargs)
        except ValueError:
            continue
        assert False, f"accepted {kwargs}"


def test_registered_variants():
    """spingarn matches the devices at zero bias and spingarn_sector over their whole sector, both on CHOLMOD in the
    nested-dissection ordering; spingarn_slu (the original SuperLU solver) and spingarn_cholmod (CHOLMOD in its own
    ordering) differ from spingarn only in the factorization. Explicit options override every default."""
    assert {name: SOLVERS[name] for name in ["spingarn", "spingarn_sector", "spingarn_slu", "spingarn_cholmod"]} == dict(
        spingarn=SpingarnSolver, spingarn_sector=SpingarnSectorSolver, spingarn_slu=SpingarnSLUSolver,
        spingarn_cholmod=SpingarnCholmodSolver)
    defaults = {cls.name: tuple(cls().options[k] for k in ("gamma", "factorization", "ordering", "blas_threads"))
                for cls in [SpingarnSolver, SpingarnSectorSolver, SpingarnSLUSolver, SpingarnCholmodSolver]}
    assert defaults == dict(spingarn=("r_lo", "cholmod", "nested_dissection", 1),
                            spingarn_sector=("geometric", "cholmod", "nested_dissection", 1),
                            spingarn_slu=("r_lo", "slu", "builtin", 1),
                            spingarn_cholmod=("r_lo", "cholmod", "builtin", 1))
    assert SpingarnSectorSolver(gamma="r_lo").options["gamma"] == "r_lo"
    assert SpingarnSLUSolver(ordering="nested_dissection").options["ordering"] == "nested_dissection"
    # the sector match has the smaller worst-case bound, the zero-bias match the fewer iterations here
    cfg = Config(4, 4, seed=0)
    a, b = SpingarnSolver().solve(cfg), SpingarnSectorSolver().solve(cfg)
    assert b.history["contraction_bound"] < a.history["contraction_bound"] and a.iterations < b.iterations
    assert np.abs(a.v - b.v).max() < 1e-8 * np.abs(a.v).max()


def test_runs_in_the_experiment_runner():
    """The variants run in one sweep, side by side, as separately named algorithms, each in a process of its own whose
    peak memory is recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        solvers = [SpingarnSolver(), SpingarnSectorSolver(), SpingarnSLUSolver()]
        records = run_sweep([(2, 2), (4, 4)], 2, solvers, tmp, save_pickles=False)
        assert [r["status"] for r in records] == ["converged"] * 12
        assert [r["algorithm"] for r in records] == ["spingarn", "spingarn_sector", "spingarn_slu"] * 4
        assert all(len(r["history"]["kcl_history"]) == r["iterations"] > 0 for r in records)
        assert all(r["peak_memory"] > 0 for r in records)
        spec = json.load(open(Path(tmp) / "experiment.json"))
        assert spec["solvers"] == [dict(name=name, tol=1e-9, max_iterations=10_000, gamma=gamma, alpha=1.0, abstol=1e-12,
                                        factorization=factorization, ordering=ordering, blas_threads=1)
                                   for name, gamma, factorization, ordering in [
                                       ("spingarn", "r_lo", "cholmod", "nested_dissection"),
                                       ("spingarn_sector", "geometric", "cholmod", "nested_dissection"),
                                       ("spingarn_slu", "r_lo", "slu", "builtin")]]
        assert (Path(tmp) / "memory.png").stat().st_size > 0


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
