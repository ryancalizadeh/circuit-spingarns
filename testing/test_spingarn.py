import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Config import Config, BranchKind
from algorithms import SOLVERS
from algorithms.spingarns import SpingarnSectorSolver, SpingarnSolver, device_resolvent, port_resistances
from run_experiments import run_sweep
from test_config import solve as newton


def test_matches_newton_reference():
    """Spingarn's answer agrees with an independent Newton solve for every port-resistance rule and relaxation, and
    satisfies the branch laws to rounding error."""
    for p, q in [(1, 1), (4, 1), (1, 5), (3, 4), (16, 32)]:
        cfg = Config(p, q, seed=p * 100 + q)
        u_ref, _ = newton(cfg)
        v_ref = cfg.incidence().T @ u_ref
        for gamma in ["geometric", "r_lo", 3e3]:
            for alpha in [1.0, 0.5]:
                out = SpingarnSolver(gamma=gamma, alpha=alpha).solve(cfg)
                assert out.converged, (p, q, gamma, alpha)
                assert np.abs(out.v - v_ref).max() < 1e-8 * np.abs(v_ref).max(), (p, q, gamma, alpha)
                assert np.abs(out.i - cfg.current(out.v)).max() < 1e-14 * np.abs(out.i).max()
                assert out.history["kvl_residual_rel"] <= 1e-9


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


def test_port_resistances():
    """Linear branches are matched exactly by every rule; only the devices' port resistance depends on it."""
    cfg = Config(3, 4, seed=0)
    d = cfg.slices[BranchKind.DEVICE]
    linear = cfg.kind != BranchKind.DEVICE
    for rule, device in [("geometric", 1e4), ("r_lo", 1e3), (2.5e3, 2.5e3)]:
        gamma = port_resistances(cfg, rule)
        assert np.all(gamma[linear] == cfg.r_hi[linear]) and np.allclose(gamma[d], device)


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
                   dict(max_iterations=0)]:
        try:
            SpingarnSolver(**kwargs)
        except ValueError:
            continue
        assert False, f"accepted {kwargs}"


def test_registered_variants_differ_only_in_device_matching():
    """spingarn matches the devices at zero bias and spingarn_sector over their whole sector; an explicit gamma
    overrides either default."""
    assert SOLVERS["spingarn"] is SpingarnSolver and SOLVERS["spingarn_sector"] is SpingarnSectorSolver
    cfg = Config(4, 4, seed=0)
    zero_bias, sector = SpingarnSolver(), SpingarnSectorSolver()
    assert zero_bias.options["gamma"] == "r_lo" and sector.options["gamma"] == "geometric"
    assert SpingarnSectorSolver(gamma="r_lo").options["gamma"] == "r_lo"
    # the sector match has the smaller worst-case bound, the zero-bias match the fewer iterations here
    a, b = zero_bias.solve(cfg), sector.solve(cfg)
    assert b.history["contraction_bound"] < a.history["contraction_bound"] and a.iterations < b.iterations
    assert np.abs(a.v - b.v).max() < 1e-8 * np.abs(a.v).max()


def test_runs_in_the_experiment_runner():
    """Both variants run in one sweep, side by side, as separately named algorithms."""
    with tempfile.TemporaryDirectory() as tmp:
        records = run_sweep([(2, 2), (4, 4)], 2, [SpingarnSolver(), SpingarnSectorSolver()], tmp, save_pickles=False)
        assert [r["status"] for r in records] == ["converged"] * 8
        assert [r["algorithm"] for r in records] == ["spingarn", "spingarn_sector"] * 4
        assert all(len(r["history"]["kcl_history"]) == r["iterations"] > 0 for r in records)
        spec = json.load(open(Path(tmp) / "experiment.json"))
        assert spec["solvers"] == [dict(name=name, tol=1e-9, max_iterations=10_000, gamma=gamma, alpha=1.0, abstol=1e-12)
                                   for name, gamma in [("spingarn", "r_lo"), ("spingarn_sector", "geometric")]]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
