import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Config import Config, BranchKind, GROUND

K = BranchKind


def device_law(cfg, v):
    """Eq. (11), written out independently of Config.current."""
    return v / cfg.R_max + (1 / cfg.R_min - 1 / cfg.R_max) * np.tanh(v)


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
    for r in range(p):
        for c in range(q):
            i_dev = device_law(cfg, W[r, c] - B[r, c])
            rw[r, c] += i_dev
            rb[r, c] -= i_dev
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
        lo, hi = (I, hi) if device_law(cfg, vd) > I else (lo, I)
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


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
