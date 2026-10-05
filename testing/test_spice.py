import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Config import Config
from algorithms.spice import SpiceSolver, kcl_check, netlist, op_method
from run_experiments import run_sweep
from test_config import solve as newton


def test_netlist_has_one_element_per_branch():
    """Every Config branch is one SPICE element, except sources, which are an ideal V plus their series R."""
    for p, q in [(1, 1), (3, 4), (5, 2)]:
        cfg = Config(p, q, seed=0)
        kinds = [line[0] for line in netlist(cfg).splitlines() if not line.startswith(".")]
        assert kinds.count("B") == p * q
        assert kinds.count("V") == p
        assert kinds.count("R") == cfg.num_branches - p * q
        assert len(kinds) == cfg.num_branches + p


def test_matches_newton_reference():
    """SPICE's operating point agrees with an independent Newton solve to rounding error."""
    solver = SpiceSolver()
    for p, q in [(1, 1), (4, 1), (1, 5), (3, 4), (16, 32)]:
        cfg = Config(p, q, seed=p * 100 + q)
        out = solver.solve(cfg)
        u_ref, _ = newton(cfg)
        v_ref = cfg.incidence().T @ u_ref
        assert out.converged and out.iterations > 0, (p, q, out.history)
        assert out.history["method"] == "newton"
        assert "KLU" in out.history["linear_solver"]
        assert np.abs(out.v - v_ref).max() < 1e-12
        assert out.history["kcl_residual_rel"] < 1e-9


def test_sparse_matrix_solver_agrees():
    cfg = Config(8, 8, seed=5)
    klu = SpiceSolver().solve(cfg)
    sparse = SpiceSolver(matrix_solver="sparse").solve(cfg)
    assert "SPARSE" in sparse.history["linear_solver"]
    assert sparse.converged and np.abs(sparse.v - klu.v).max() < 1e-12


def test_options_reach_ngspice():
    s = SpiceSolver(tol=1e-6, max_iterations=500, abstol=1e-15, gminsteps=0, noopiter=True)
    assert s.spice_options() == ["reltol=1e-06", "itl1=500", "klu", "abstol=1e-15", "gminsteps=0", "noopiter"]
    # noopiter skips plain Newton, gminsteps=0 then leaves source stepping to find the operating point
    out = s.solve(Config(4, 4, seed=0))
    assert out.converged and out.history["method"] == "source stepping"


def test_kcl_check_rejects_a_perturbed_solution():
    cfg = Config(6, 6, seed=2)
    u, _ = newton(cfg)
    assert kcl_check(cfg, u, reltol=1e-3, abstol=1e-12)[2]
    u_bad = u.copy()
    u_bad[cfg.bit_node(2, 3)] += 1e-3  # 1 mV off at a single node
    _, _, ok, _, rel = kcl_check(cfg, u_bad, reltol=1e-3, abstol=1e-12)
    assert not ok and rel > 1e-3


def test_op_method_from_ngspice_notes():
    """Notes as ngspice 47 prints them for each way of reaching the operating point."""
    assert op_method([]) == "newton"
    assert op_method(["Note: Starting dynamic gmin stepping", "Note: Dynamic gmin stepping completed"]) \
        == "gmin stepping"
    assert op_method(["Note: Starting source stepping", "Note: Source stepping completed"]) == "source stepping"
    assert op_method(["Note: Starting dynamic gmin stepping", "Warning: Dynamic gmin stepping failed",
                      "Note: Starting true gmin stepping", "Warning: True gmin stepping failed",
                      "Note: Starting source stepping", "Warning: source stepping failed",
                      "Note: Transient op started", "Note: Transient op finished successfully"]) == "transient op"


def test_runs_in_the_experiment_runner():
    with tempfile.TemporaryDirectory() as tmp:
        records = run_sweep([(2, 2), (4, 4)], 2, [SpiceSolver()], tmp, save_pickles=False)
        assert [r["status"] for r in records] == ["converged"] * 4
        assert all(r["iterations"] > 0 and r["history"]["kcl_residual_rel"] < 1e-9 for r in records)
        spec = json.load(open(Path(tmp) / "experiment.json"))
        assert spec["solvers"] == [dict(name="spice", reltol=1e-3, itl1=100, itl1_effective=100, matrix_solver="klu")]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
