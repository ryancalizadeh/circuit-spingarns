import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Config import Config, BranchKind, GROUND

K = BranchKind


def device_law(cfg, v):
    """Eq. (11) for every cell with its own R_min and R_max, applied to max(v, 0) in a 1D1R cell; for a 1S1R cell the
    series law by bisection. Written out independently of Config.current and devices.py."""
    if cfg.device == "1s1r":
        return series_law_bisect(cfg, v)
    if cfg.device == "1d1r":
        v = np.where(v > 0, v, 0.0)
    return v / cfg.R_max_cell + (1 / cfg.R_min_cell - 1 / cfg.R_max_cell) * np.tanh(v)


def selector_law(cfg, v):
    """The 1S1R selector, piecewise: conductance 1/R_off below V_th - delta, 1/R_on above V_th + delta, ramping linearly
    in between, odd in v."""
    def ramp(x):  # integral of the conductance ramp from 0 to 1 centred on x = 0
        if cfg.delta == 0:
            return np.where(x > 0, x, 0.0)
        return np.select([x <= -cfg.delta, x >= cfg.delta], [0.0, x], (x + cfg.delta) ** 2 / (4 * cfg.delta))
    return v / cfg.R_off + (1 / cfg.R_on - 1 / cfg.R_off) * (ramp(v - cfg.V_th) - ramp(-v - cfg.V_th))


def series_law_bisect(cfg, v):
    """1S1R cell currents at cell voltages v (one per cell), by nested bisection: the selector voltage x solves
    x + g_m^-1(g_s(x)) = v, increasing in x."""
    g_m = lambda x: x / cfg.R_max_cell + (1 / cfg.R_min_cell - 1 / cfg.R_max_cell) * np.tanh(x)

    def bisect(f, lo, hi):  # root of the increasing f in [lo, hi], elementwise
        for _ in range(80):
            mid = (lo + hi) / 2
            below = f(mid) < 0
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo + hi) / 2

    def g_m_inv(i):
        bound = cfg.R_max_cell * np.abs(i)  # g_m has slope >= 1 / R_max
        return bisect(lambda x: g_m(x) - i, -bound, bound)

    x = bisect(lambda x: x + g_m_inv(selector_law(cfg, x)) - v, -np.abs(v), np.abs(v))
    return selector_law(cfg, x)


def solve(cfg, tol=1e-13):
    """Newton on nodal KCL: A g(A^T u) = 0. Returns node potentials and iteration count."""
    A = cfg.incidence()
    u = np.zeros(cfg.num_nodes)
    for it in range(100):
        v = A.T @ u
        F = A @ cfg.current(v)
        if np.max(np.abs(F)) < tol:
            return u, it
        J = (A @ (A.T.multiply(cfg.conductance(v)[:, None]))).tocsc()
        u -= spla.spsolve(J, F)
    raise RuntimeError("newton did not converge")


def kcl_from_prose(cfg, u):
    """KCL residual at every node, written straight from the paper's description (no tail/head arrays)."""
    p, q = cfg.p, cfg.q
    W = u[: p * q].reshape(p, q)
    B = u[p * q:].reshape(p, q)
    rw, rb = np.zeros((p, q)), np.zeros((p, q))  # net current leaving each node
    I = device_law(cfg, (W - B).ravel()).reshape(p, q)  # device (r, c) carries I[r, c] from w[r, c] to b[r, c]
    for r in range(p):
        for c in range(q):
            rw[r, c] += I[r, c]
            rb[r, c] -= I[r, c]
            if c + 1 < q:
                rw[r, c] += (W[r, c] - W[r, c + 1]) / cfg.R_wire
            if c > 0:
                rw[r, c] += (W[r, c] - W[r, c - 1]) / cfg.R_wire
            if c == 0:  # source E_r with R_s from ground drives the head of the word line
                rw[r, c] += (W[r, c] - cfg.E[r]) / cfg.R_source
            if r + 1 < p:
                rb[r, c] += (B[r, c] - B[r + 1, c]) / cfg.R_wire
            if r > 0:
                rb[r, c] += (B[r, c] - B[r - 1, c]) / cfg.R_wire
            if r == p - 1:  # load at the foot of the bit line
                rb[r, c] += B[r, c] / cfg.R_load
    return np.concatenate([rw.ravel(), rb.ravel()])


def test_counts_kinds_and_incidence():
    """2pq nodes, m = 3pq branches split p(q-1), q(p-1), pq, p, q; full-rank reduced incidence."""
    for p, q in [(1, 1), (1, 5), (4, 1), (3, 4), (7, 2)]:
        cfg = Config(p, q, seed=0)
        sizes = {k: cfg.slices[k].stop - cfg.slices[k].start for k in K}
        expect = {K.WORD_WIRE: p * (q - 1), K.BIT_WIRE: q * (p - 1), K.DEVICE: p * q, K.SOURCE: p, K.LOAD: q}
        assert sizes == expect, (p, q, sizes)
        assert cfg.num_nodes == 2 * p * q and cfg.num_branches == 3 * p * q == len(cfg.kind)
        for k in K:
            assert np.all(cfg.kind[cfg.slices[k]] == k)
        A = cfg.incidence()
        colsum = np.asarray(A.sum(axis=0)).ravel()
        to_ground = cfg.head == GROUND
        assert np.all(colsum[to_ground] == 1) and np.all(colsum[~to_ground] == 0)
        assert np.linalg.matrix_rank(A.toarray()) == cfg.num_nodes


def test_device_law_and_sector_bounds():
    cfg = Config(2, 3, seed=1)
    v = np.linspace(-3, 3, cfg.num_branches)
    d = cfg.slices[K.DEVICE]
    assert np.allclose(cfg.current(v)[d], device_law(cfg, v[d]), rtol=1e-14)
    for row in np.random.default_rng(0).normal(scale=5, size=(1000, cfg.num_branches)):
        g = cfg.conductance(row)
        assert np.all(g >= 1 / cfg.r_hi - 1e-18) and np.all(g <= 1 / cfg.r_lo + 1e-18)
    h = 1e-6
    fd = (cfg.current(v + h) - cfg.current(v - h)) / (2 * h)
    assert np.allclose(fd, cfg.conductance(v), rtol=1e-6)


def test_single_cell_matches_series_loop():
    """A 1x1 crossbar is the series loop E - R_s - device - R_L; compare with bisection on the loop current."""
    cfg = Config(1, 1, seed=3)
    u, _ = solve(cfg)
    E = cfg.E[0]
    lo, hi = 0.0, E / (cfg.R_source + cfg.R_load)
    for _ in range(200):
        I = (lo + hi) / 2
        vd = E - I * (cfg.R_source + cfg.R_load)  # voltage left for the device
        lo, hi = (I, hi) if device_law(cfg, np.array([vd]))[0] > I else (lo, I)
    I_series = (lo + hi) / 2
    i = cfg.current(cfg.incidence().T @ u)
    assert np.isclose(i[cfg.slices[K.DEVICE]][0], I_series, rtol=1e-10)
    assert np.isclose(-i[cfg.slices[K.SOURCE]][0], I_series, rtol=1e-10)  # source delivers the loop current
    assert np.isclose(i[cfg.slices[K.LOAD]][0], I_series, rtol=1e-10)


def test_solution_satisfies_prose_kcl():
    """Config's nodal solution satisfies KCL written from the prose, is physical, and balances power."""
    for p, q in [(3, 4), (8, 8), (16, 32)]:
        cfg = Config(p, q, seed=p * 100 + q)
        u, _ = solve(cfg)
        scale = cfg.E.max() / cfg.R_source
        assert np.abs(kcl_from_prose(cfg, u)).max() < 1e-9 * scale
        assert np.all(u > 0) and u.max() < cfg.E.max()
        v = cfg.incidence().T @ u
        assert abs(v @ cfg.current(v)) < 1e-12  # Tellegen


RANDOM_CELLS = dict(R_min=(1e3, 1e4), R_max=(5e4, 2e5))


def test_randomized_cells():
    """Per-cell R_min and R_max are drawn log-uniformly within their ranges, reproducibly, after E, so a seed keeps its
    sources; params() regenerates the same cells; fixed values give every cell exactly that value."""
    fixed, rand = Config(20, 30, seed=7), Config(20, 30, seed=7, **RANDOM_CELLS)
    assert np.array_equal(fixed.E, rand.E)
    assert np.all(fixed.R_min_cell == 1e3) and np.all(fixed.R_max_cell == 100e3)
    for cells, (lo, hi) in [(rand.R_min_cell, (1e3, 1e4)), (rand.R_max_cell, (5e4, 2e5))]:
        assert cells.shape == (600,) and np.all((cells >= lo) & (cells <= hi))
        assert abs(np.median(np.log(cells)) - np.log(np.sqrt(lo * hi))) < 0.1 * np.log(hi / lo)  # log-uniform
    again = Config(**rand.params())
    assert np.array_equal(again.R_min_cell, rand.R_min_cell) and np.array_equal(again.R_max_cell, rand.R_max_cell)
    assert not np.array_equal(Config(20, 30, seed=8, **RANDOM_CELLS).R_min_cell, rand.R_min_cell)
    d = rand.slices[K.DEVICE]
    assert np.array_equal(rand.r_lo[d], rand.R_min_cell) and np.array_equal(rand.r_hi[d], rand.R_max_cell)
    for bad in [dict(R_min=(1e3, 6e4), R_max=(5e4, 2e5)), dict(R_min=(1e4, 1e3)), dict(R_min=0.0),
                dict(device="diode")]:
        try:
            Config(2, 2, seed=0, **bad)
        except ValueError:
            continue
        assert False, f"accepted {bad}"


def test_1d1r_law_and_sector():
    """A 1D1R cell carries no current for v <= 0 and its memristor's current for v > 0; its incremental conductance is
    0 in reverse and the memristor's forward, so its sector is [R_min, inf]."""
    cfg = Config(3, 4, seed=2, device="1d1r", **RANDOM_CELLS)
    mem = Config(3, 4, seed=2, **RANDOM_CELLS)
    d = cfg.slices[K.DEVICE]
    assert np.array_equal(cfg.r_lo[d], cfg.R_min_cell) and np.all(np.isinf(cfg.r_hi[d]))
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.normal(scale=2, size=cfg.num_branches)
        i, g = cfg.current(v), cfg.conductance(v)
        fwd = v[d] > 0
        assert np.all(i[d][~fwd] == 0) and np.array_equal(i[d][fwd], mem.current(v)[d][fwd])
        linear = cfg.kind != K.DEVICE
        assert np.array_equal(i[linear], mem.current(v)[linear])
        assert np.all(g >= 1 / cfg.r_hi) and np.all(g <= 1 / cfg.r_lo * (1 + 1e-15))
        assert np.all(g[d][~fwd] == 0) and np.array_equal(g[d][fwd], mem.conductance(v)[d][fwd])
    assert np.all(cfg.current(np.zeros(cfg.num_branches))[d] == 0)


def test_1d1r_solution_satisfies_prose_kcl():
    """Newton's nodal solution of 1D1R crossbars with signed inputs satisfies KCL written from the prose, blocks current
    in every reverse-biased cell, conducts in forward-biased ones and balances power, with both kinds present."""
    for p, q in [(1, 1), (3, 4), (8, 8), (16, 32)]:
        cfg = Config(p, q, seed=p * 100 + q, device="1d1r", E_range=(-1, 1.5), **RANDOM_CELLS)
        u, _ = solve(cfg)
        scale = np.abs(cfg.E).max() / cfg.R_source
        assert np.abs(kcl_from_prose(cfg, u)).max() < 1e-9 * scale
        v = cfg.incidence().T @ u
        i = cfg.current(v)
        d = cfg.slices[K.DEVICE]
        assert np.all(i[d][v[d] <= 0] == 0) and np.all(i[d][v[d] > 0] > 0)
        if p * q > 1:
            assert 0 < np.mean(v[d] <= 0) < 1
        assert abs(v @ i) < 1e-12  # Tellegen


SELECTOR = dict(device="1s1r", E_range=(-1, 1.5), R_on=1e3, R_off=1e6, V_th=0.5, **RANDOM_CELLS)


def test_selector_law():
    """devices.selector_current is the piecewise selector: odd, conductance exactly 1/R_off below V_th - delta and
    1/R_on above V_th + delta, within them in between, matching finite differences; delta -> 0 approaches the kink."""
    from devices import selector_conductance, selector_current
    v = np.linspace(-2, 2, 40001)
    for delta in (0.1, 0.01, 0.0):
        cfg = Config(1, 1, seed=0, **{**SELECTOR, "delta": delta})
        args = (cfg.R_on, cfg.R_off, cfg.V_th, delta)
        i, g = selector_current(v, *args), selector_conductance(v, *args)
        assert np.allclose(i, selector_law(cfg, v), rtol=1e-13, atol=1e-18)
        assert np.allclose(selector_current(-v, *args), -i, rtol=1e-13, atol=1e-18)
        assert np.all(np.diff(i) > 0)
        assert np.all(g >= 1 / cfg.R_off - 1e-18) and np.all(g <= 1 / cfg.R_on + 1e-18)
        off, on = np.abs(v) <= cfg.V_th - delta, np.abs(v) >= cfg.V_th + delta + 1e-12
        assert np.allclose(g[off], 1 / cfg.R_off, rtol=1e-12) and np.allclose(g[on], 1 / cfg.R_on, rtol=1e-12)
        if delta > 0:  # the conductance has kinks at |v| = V_th +- delta, where central differences are off by O(h)
            h = 1e-7
            fd = (selector_current(v + h, *args) - selector_current(v - h, *args)) / (2 * h)
            smooth = np.min(np.abs(np.abs(v)[:, None] - [cfg.V_th - delta, cfg.V_th + delta]), axis=1) > 2 * h
            assert np.allclose(fd[smooth], g[smooth], rtol=1e-5, atol=1e-12)
    sharp, kink = Config(1, 1, seed=0, **{**SELECTOR, "delta": 1e-6}), Config(1, 1, seed=0, **{**SELECTOR, "delta": 0})
    assert np.abs(selector_law(sharp, v) - selector_law(kink, v)).max() < 1e-9


def test_1s1r_cell_law_and_sector():
    """A 1S1R cell's current is the series law (checked against nested bisection); series_solve's split satisfies
    v = v_s + v_m and g_s(v_s) = g_m(v_m) = i; the conductance is the series combination, matches finite differences,
    and stays in the sector [R_on + R_min, R_off + R_max]."""
    from devices import memristor_current, selector_current
    for delta in (0.05, 0.0):
        cfg = Config(4, 5, seed=3, **{**SELECTOR, "delta": delta})
        d = cfg.slices[K.DEVICE]
        assert np.array_equal(cfg.r_lo[d], cfg.R_on + cfg.R_min_cell)
        assert np.array_equal(cfg.r_hi[d], cfg.R_off + cfg.R_max_cell)
        rng = np.random.default_rng(1)
        for scale in (0.3, 1.0, 3.0):
            v = rng.normal(scale=scale, size=cfg.p * cfg.q)
            i = cfg.device_current(v)
            assert np.allclose(i, series_law_bisect(cfg, v), rtol=1e-9, atol=1e-16)
            v_cell, i2, v_s, v_m, _ = cfg.series_solve(v)
            assert np.allclose(v_cell, v, rtol=1e-12, atol=1e-15) and np.array_equal(i, i2)
            assert np.allclose(selector_current(v_s, cfg.R_on, cfg.R_off, cfg.V_th, delta), i, rtol=1e-12, atol=1e-18)
            assert np.allclose(memristor_current(v_m, cfg.R_min_cell, cfg.R_max_cell), i, rtol=1e-11, atol=1e-18)
            g = cfg.device_conductance(v)
            assert np.all(g >= 1 / cfg.r_hi[d] * (1 - 1e-12)) and np.all(g <= 1 / cfg.r_lo[d] * (1 + 1e-12))
            if delta > 0:
                h = 1e-7
                fd = (cfg.device_current(v + h) - cfg.device_current(v - h)) / (2 * h)
                assert np.allclose(fd, g, rtol=1e-5, atol=1e-12)
        params = cfg.params()
        assert (params["R_on"], params["R_off"], params["V_th"], params["delta"]) == (1e3, 1e6, 0.5, delta)
        assert np.array_equal(Config(**params).device_current(v), i)
    for bad in [dict(R_on=2e6), dict(delta=0.5), dict(delta=-0.01)]:
        try:
            Config(2, 2, seed=0, **{**SELECTOR, **bad})
        except ValueError:
            continue
        assert False, f"accepted {bad}"


def test_1s1r_solution_satisfies_prose_kcl():
    """Newton's nodal solution of 1S1R crossbars, with a rounded knee and with the kink, satisfies KCL written from the
    prose and balances power, with both conducting and blocking selectors present."""
    for delta in (0.05, 0.0):
        for p, q in [(1, 1), (3, 4), (8, 8)]:
            cfg = Config(p, q, seed=p * 100 + q, **{**SELECTOR, "delta": delta})
            u, _ = solve(cfg)
            scale = np.abs(cfg.E).max() / cfg.R_source
            assert np.abs(kcl_from_prose(cfg, u)).max() < 1e-9 * scale
            v = cfg.incidence().T @ u
            assert abs(v @ cfg.current(v)) < 1e-12  # Tellegen
            if p * q > 1:
                _, _, v_s, _, _ = cfg.series_solve(v[cfg.slices[K.DEVICE]])
                on = np.abs(v_s) > cfg.V_th
                assert 0 < np.mean(on) < 1, (p, q, delta)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
