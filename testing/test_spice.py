import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Config import Config
from Config import BranchKind
from algorithms.spice import (SpiceExpSolver, SpiceSolver, kcl_check, netlist, ngspice, op_method, read_op,
                              spice_diode_current)
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


def test_1d1r_matches_newton_reference():
    """1D1R cells (uramp in the behavioral source) with per-cell resistances and signed inputs: SPICE's operating point
    agrees with Newton's."""
    solver = SpiceSolver()
    for p, q in [(1, 1), (3, 4), (16, 32)]:
        cfg = Config(p, q, seed=p * 100 + q, device="1d1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
        assert netlist(cfg).count("uramp(") == 2 * p * q
        out = solver.solve(cfg)
        u_ref, _ = newton(cfg)
        v_ref = cfg.incidence().T @ u_ref
        assert out.converged, (p, q, out.history)
        assert np.abs(out.v - v_ref).max() < 1e-8, (p, q)


def test_1s1r_matches_newton_reference():
    """1S1R cells, each a selector and a memristor in series through an internal node, with a rounded knee and with the
    kink: SPICE's operating point agrees with Newton's."""
    solver = SpiceSolver()
    for delta in (0.05, 0.0):
        for p, q in [(1, 1), (3, 4), (8, 8)]:
            cfg = Config(p, q, seed=p * 100 + q, device="1s1r", E_range=(-1, 1.5), R_min=(1e3, 1e4),
                         R_max=(5e4, 2e5), R_on=1e3, R_off=1e6, V_th=0.5, delta=delta)
            text = netlist(cfg)
            assert sum(line.startswith("BS") for line in text.splitlines()) == p * q
            assert sum(line.startswith("B") for line in text.splitlines()) == 2 * p * q
            out = solver.solve(cfg)
            u_ref, _ = newton(cfg)
            v_ref = cfg.incidence().T @ u_ref
            assert out.converged, (p, q, delta, out.history)
            assert np.abs(out.v - v_ref).max() < 1e-8, (p, q, delta)


def single_op(text):
    """Operating point of a small netlist, by name."""
    from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError
    ng = ngspice()
    before = set(ng.plot_names)
    raw = Path(tempfile.gettempdir()) / "test_spice_single_op.raw"
    try:
        ng.load_circuit(text)
        try:
            ng.exec_command("run")
        except NgSpiceCommandError:
            pass
        plot = next(name for name in ng.plot_names if name.startswith("op") and name not in before)
        return read_op(ng, plot, raw)
    finally:
        for command in ("destroy all", "remcirc"):
            try:
                ng.exec_command(command)
            except NgSpiceCommandError:
                pass
        raw.unlink(missing_ok=True)


def test_diode_element_law_and_exp_clamp():
    """ngspice's diode element, with N scaled as netlist() does, follows spice_diode_current: the Shockley law down to
    -3 n V_T, SPICE3's cubic reverse-bias approximation below (up to 0.4% of I_s off the exponential), plus gmin across
    the junction. Behavioral sources clamp exp() at 1e99, flat beyond an argument of 227.96."""
    from algorithms.spice import VT_NGSPICE
    I_s, n, V_T = 1e-9, 1.3, 0.02585
    c = n * V_T
    for gmin in (1e-12, 1e-20):
        for v in (0.4, 0.05, -0.02, -3 * c, -0.2, -1.0):
            out = single_op(f".title d\nV1 a 0 {v!r}\nD1 a 0 dm\n.model dm D(IS={I_s!r} N={n * V_T / VT_NGSPICE!r} RS=0)\n"
                            f".options gmin={gmin!r}\n.op\n.end")
            expect = spice_diode_current(np.array(v), I_s, c, gmin)
            assert abs(-out["i(v1)"] - expect) <= 1e-9 * abs(expect) + 1e-22, (gmin, v, -out["i(v1)"], expect)
    assert abs(spice_diode_current(np.array(-0.2), I_s, c, 0.0) / (I_s * np.expm1(-0.2 / c)) - 1) > 1e-3
    for x, expect in [(100, np.exp(100)), (227, np.exp(227)), (228, 1e99), (800, 1e99)]:
        out = single_op(f".title b\nV1 a 0 {x}\nB1 b 0 I=exp(V(a))\nR1 b 0 1\n.op\n.end")
        assert np.isclose(-out["b"], expect, rtol=1e-6), (x, -out["b"])


SHOCKLEY = dict(device="1d1r_shockley", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))


def test_internal_node_netlists():
    """A Shockley diode on its own branch is one D element (with one shared .model) or one behavioral exp source, an
    ideal one a uramp source with the solver's on-resistance; each cell's memristor runs from its internal node. The lumped
    Shockley cell has no netlist."""
    p, q = 3, 4
    cfg = Config(p, q, seed=0, internal_node=True, **SHOCKLEY)
    for diode, element in (("D", "D"), ("exp", "BD")):
        lines = netlist(cfg, diode=diode).splitlines()
        assert sum(line.split()[0].startswith(element) for line in lines if not line.startswith(".")) == p * q
        assert sum(line.startswith("B") and "tanh(V(m" in line for line in lines) == p * q
    assert netlist(cfg, diode="D").count(".model") == 1 and "exp(" in netlist(cfg, diode="exp")
    ideal = Config(p, q, seed=0, internal_node=True, device="1d1r", E_range=(-1, 1.5))
    text = netlist(ideal, ron=1e-3)
    assert text.count("uramp(V(w") == p * q and "1000.0*uramp" in text
    for bad in (lambda: netlist(Config(2, 2, seed=0, **SHOCKLEY)), lambda: netlist(cfg, diode="ideal"),
                lambda: SpiceSolver(diode="exp2"), lambda: SpiceSolver(ron=0.0)):
        try:
            bad()
        except ValueError:
            continue
        assert False, "accepted a bad diode setting"


def test_internal_node_diodes_match_newton_reference():
    """SPICE on internal-node 1D1R crossbars agrees at the word and bit nodes with Newton on the lumped cell: Shockley
    diodes as D elements (to SPICE's tolerances and its reverse-bias law) and as behavioral exp sources, ideal diodes
    regularized by an
    on-resistance (to an error that shrinks with it). Convergence is judged on the laws ngspice solved, and the residual
    under the exact laws is recorded beside it."""
    for p, q in [(1, 1), (3, 4), (8, 8)]:
        for I_s in (1e-9, 1e-6):
            ref = Config(p, q, seed=p * 100 + q, I_s=I_s, **SHOCKLEY)
            u_ref = newton(ref)[0][:2 * p * q]
            cfg = Config(p, q, seed=p * 100 + q, I_s=I_s, internal_node=True, **SHOCKLEY)
            # tolerances: SPICE stops on its voltage test (RELTOL 1e-3, VNTOL 1 uV); and the D element's cubic
            # reverse-bias law shifts each reverse cell's current by up to 0.4% of I_s, 4 nA at I_s = 1e-6 A, which
            # a row of such cells drops across its 50 ohm source (~6 uV at 8x8)
            for solver, tol in [(SpiceSolver(), 2e-5), (SpiceExpSolver(), 2e-5)]:
                out = solver.solve(cfg)
                h = out.history
                assert out.converged, (p, q, I_s, solver.name, h["method"], h["notes"][:3])
                assert np.abs(cfg.line_potentials(out.v) - u_ref).max() < tol, (p, q, I_s, solver.name)
                assert ("kcl_exact_residual" in h) == (solver.name == "spice")
        lumped = Config(p, q, seed=p * 100 + q, device="1d1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5))
        u_ref = newton(lumped)[0][:2 * p * q]
        ideal = Config(p, q, seed=p * 100 + q, device="1d1r", E_range=(-1, 1.5), R_min=(1e3, 1e4), R_max=(5e4, 2e5),
                       internal_node=True)
        errors = []
        for ron in (1.0, 1e-3):
            out = SpiceSolver(ron=ron).solve(ideal)
            assert out.converged and out.history["kcl_exact_residual"] >= 0, (p, q, ron)
            e = ideal.slices[BranchKind.DIODE]
            assert np.allclose(out.i[e], np.maximum(out.v[e], 0) / ron, rtol=1e-12, atol=1e-18)
            errors.append(np.abs(ideal.line_potentials(out.v) - u_ref).max())
        assert errors[1] < 1e-6 and errors[1] < errors[0] / 100, (p, q, errors)


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


def test_kcl_check_rejects_overflowed_currents():
    """Potentials far enough out to overflow a branch law fail the KCL test, although inf <= reltol * inf."""
    cfg = Config(2, 2, seed=0, device="1d1r_shockley", internal_node=True, I_s=1e-9)
    u = np.zeros(cfg.num_nodes)
    u[0] = 4e71
    with np.errstate(all="ignore"):
        assert not kcl_check(cfg, u, reltol=1e-3, abstol=1e-12)[2]


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
    # every fallback failing, as at 1024x1024 with KLU
    assert op_method(["Warning: singular matrix:  check node 0", "Note: Starting dynamic gmin stepping",
                      "Warning: Dynamic gmin stepping failed", "Note: Starting true gmin stepping",
                      "Warning: True gmin stepping failed", "Note: Starting source stepping",
                      "Warning: source stepping failed", "Note: Transient op started",
                      "Error: Transient op failed, timestep too small",
                      "Error: The operating point could not be simulated successfully."]) == "failed"


def test_runs_in_the_experiment_runner():
    with tempfile.TemporaryDirectory() as tmp:
        records = run_sweep([(2, 2), (4, 4)], 2, [SpiceSolver()], tmp, save_pickles=False)
        assert [r["status"] for r in records] == ["converged"] * 4
        assert all(r["iterations"] > 0 and r["history"]["kcl_residual_rel"] < 1e-9 for r in records)
        spec = json.load(open(Path(tmp) / "experiment.json"))
        assert spec["solvers"] == [dict(name="spice", reltol=1e-3, itl1=100, itl1_effective=100, matrix_solver="klu",
                                        diode="D", ron=1e-3)]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
