from enum import IntEnum

import numpy as np
import scipy.sparse as sp

from devices import (diode_series_solve, memristor_conductance, memristor_current, selector_conductance,
                     series_solve, shockley_conductance, shockley_current)

GROUND = -1
DEVICES = ("memristor", "1d1r", "1s1r", "1d1r_shockley")
DIODE_DEVICES = ("1d1r", "1d1r_shockley")


def _range(value):
    """A resistance given as one value or as a (lo, hi) range, as a (lo, hi) pair of floats."""
    lo, hi = (value, value) if isinstance(value, (int, float)) else value
    return float(lo), float(hi)


def _log_uniform(rng, bounds, n):
    """n values drawn log-uniformly from bounds = (lo, hi); exactly lo, and no draw, if lo == hi."""
    lo, hi = bounds
    if lo == hi:
        return np.full(n, lo)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


class BranchKind(IntEnum):
    """Branch kinds, in the order their blocks appear in the branch numbering."""
    WORD_WIRE = 0
    BIT_WIRE = 1
    DEVICE = 2
    SOURCE = 3
    LOAD = 4
    DIODE = 5  # diodes on branches of their own (internal_node=True); an empty block otherwise


class Config:
    """
    A configuration describing a memristive crossbar array with p word lines and q bit lines.

    Word line r carries the nodes w[r, 0..q-1] joined by wire resistances R_wire, and bit line c carries
    the nodes b[0..p-1, c] joined by the same. A memristive device joins w[r, c] to b[r, c] at every
    cross-point. Each word line is driven at its head w[r, 0] by a source E[r] with series resistance
    R_source, and each bit line is terminated at its foot b[p-1, c] by a load R_load to ground.
    Every memristor is a static nonlinear resistor (a memristor read at a fixed internal state) with characteristic

        i = g_m(v) = v / R_max + (1 / R_min - 1 / R_max) * tanh(v)

    which lies in the sector S[R_min, R_max] on incremental resistance. R_min and R_max are either one value for every
    cell or a (lo, hi) range, from which each cell draws its own value log-uniformly; the per-cell values are R_min_cell
    and R_max_cell. The ranges may not overlap, so R_min <= R_max in every cell.

    `device` picks what sits at each cross-point:

        "memristor"  the memristor alone: i = g_m(v);
        "1d1r"       an ideal diode in series with the memristor: i = g_m(max(v, 0)). A cell conducts only when its
                     word node is above its bit node; reverse-biased it is an open circuit. The law is monotone but
                     flat for v < 0, with a kink at v = 0, so its incremental resistance ranges over [R_min, inf].
        "1s1r"       a threshold selector in series with the memristor (see devices.py). The selector conducts with
                     incremental resistance R_off for |v_s| <= V_th - delta and R_on for |v_s| >= V_th + delta, its
                     conductance ramping linearly in between (delta = 0: a kink at |v_s| = V_th). The cell carries
                     i = g_s(v_s) = g_m(v_m) with v = v_s + v_m, and its incremental resistance ranges over
                     [R_on + R_min, R_off + R_max]: finite, so the cell is strongly monotone however sharp the knee.
                     The selector's R_on, R_off, V_th and delta are the same for every cell.
        "1d1r_shockley"  a Shockley diode i = I_s (e^{v_d / (n V_T)} - 1) in series with the memristor, treated as one
                     branch (a "lumped" cell): i = g_d(v_d) = g_m(v_m) with v = v_d + v_m (devices.diode_series_solve).
                     The diode's incremental resistance n V_T / (i + I_s) runs from ~0 to infinity, so the cell's sector
                     is [R_min, inf), like the ideal 1D1R cell's. I_s, n and V_T are the same for every cell.

    internal_node=True (1D1R cells only) puts each cell's diode on a branch of its own: from w[r, c] to a new internal node
    m[r, c], with the memristor on from m[r, c] to b[r, c]. The diode is the ideal one for device "1d1r" and the Shockley
    one for "1d1r_shockley"; the physical circuit is the same as the lumped cell's, only the formulation differs. The
    memristors keep the DEVICE block and the diodes form the DIODE block, of sector [0, inf): the Shockley diode's
    incremental resistance ranges over (0, inf), the ideal diode's is 0 (conducting) or inf (blocking). An ideal diode on
    its own branch is set-valued (v <= 0, i >= 0, v i = 0), not a function of its voltage, so current() gives both
    branches of such a cell the lumped cell's current g_m(max(v_d + v_m, 0)), which is exact at a solution, and
    conductance() refuses: a conducting ideal diode has no finite conductance.

    Indices are zero-based. Ground is the reference node and is not numbered, which leaves 2pq nodes:
    word nodes first, then bit nodes, each row-major over (r, c), and with internal_node pq internal nodes after them
    (3pq in all). The m = 3pq branches are numbered in blocks of p(q-1) word wires, q(p-1) bit wires, pq devices, p
    sources and q loads, each block row-major over (r, c), and with internal_node pq diodes after them (4pq in all);
    `slices` maps each BranchKind to its block (DIODE's is empty without internal_node).

    Branch k runs from node tail[k] to node head[k] with associated reference directions:
    v[k] = u[tail[k]] - u[head[k]] and i[k] flows from tail to head through the branch, so v[k] * i[k]
    is the power it absorbs. Devices run from w[r, c] to b[r, c] (with internal_node from m[r, c] to b[r, c], and diodes
    from w[r, c] to m[r, c]); sources and loads run to ground.

    An object of this class is used to initialize a simulation study.
    """

    def __init__(self, num_word_lines, num_bit_lines, R_min=1e3, R_max=100e3, R_wire=5.0,
                 R_source=50.0, R_load=1e3, E_range=(0.1, 0.5), seed=None, device="memristor", R_on=1e3, R_off=1e6,
                 V_th=0.5, delta=0.01, I_s=1e-14, n=1.0, V_T=0.02585, internal_node=False):
        if num_word_lines < 1 or num_bit_lines < 1:
            raise ValueError("need at least one word line and one bit line")
        (min_lo, min_hi), (max_lo, max_hi) = _range(R_min), _range(R_max)
        if not 0 < min_lo <= min_hi <= max_lo <= max_hi:
            raise ValueError("need 0 < R_min <= R_max, with R_min's range wholly below R_max's")
        if min(R_wire, R_source, R_load) <= 0:
            raise ValueError("wire, source and load resistances must be positive")
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
        if device == "1s1r" and not (0 < R_on <= R_off and 0 <= delta < V_th):
            raise ValueError("a 1S1R selector needs 0 < R_on <= R_off and 0 <= delta < V_th")
        if device == "1d1r_shockley" and not min(I_s, n, V_T) > 0:
            raise ValueError("a Shockley diode needs I_s, n and V_T > 0")
        if internal_node and device not in DIODE_DEVICES:
            raise ValueError(f"internal_node needs a 1D1R cell, one of {DIODE_DEVICES}, not {device!r}")

        self.p = p = num_word_lines
        self.q = q = num_bit_lines
        self.R_min, self.R_max = R_min, R_max
        self.R_wire, self.R_source, self.R_load = R_wire, R_source, R_load
        self.E_range, self.seed, self.device = tuple(E_range), seed, device
        self.R_on, self.R_off, self.V_th, self.delta = R_on, R_off, V_th, delta
        self.I_s, self.n, self.V_T = I_s, n, V_T
        self.internal_node = bool(internal_node)
        # the law of the DEVICE block's branches, and of the DIODE block's (None: no such block)
        self.device_law = "memristor" if self.internal_node else device
        self.diode_law = {"1d1r": "ideal", "1d1r_shockley": "shockley"}[device] if self.internal_node else None
        rng = np.random.default_rng(seed)
        self.E = rng.uniform(*E_range, size=p)
        # Drawn after E, so a circuit with fixed R_min and R_max is exactly the one its seed always gave.
        self.R_min_cell = _log_uniform(rng, (min_lo, min_hi), p * q)
        self.R_max_cell = _log_uniform(rng, (max_lo, max_hi), p * q)

        self.num_nodes = (3 if self.internal_node else 2) * p * q
        self.num_branches = (4 if self.internal_node else 3) * p * q

        r, c = np.indices((p, q))
        w, b = self.word_node(r, c), self.bit_node(r, c)
        m = self.mid_node(r, c) if self.internal_node else np.zeros((0, 0), dtype=int)
        blocks = [
            (BranchKind.WORD_WIRE, w[:, :-1], w[:, 1:], R_wire, R_wire),
            (BranchKind.BIT_WIRE, b[:-1, :], b[1:, :], R_wire, R_wire),
            (BranchKind.DEVICE, m if self.internal_node else w, b,
             *{"memristor": (self.R_min_cell, self.R_max_cell),
               "1d1r": (self.R_min_cell, np.inf),
               "1d1r_shockley": (self.R_min_cell, np.inf),
               "1s1r": (R_on + self.R_min_cell, R_off + self.R_max_cell)}[self.device_law]),
            (BranchKind.SOURCE, w[:, 0], np.full(p, GROUND), R_source, R_source),
            (BranchKind.LOAD, b[-1, :], np.full(q, GROUND), R_load, R_load),
            (BranchKind.DIODE, w if self.internal_node else m, m, 0.0, np.inf),
        ]

        self.slices = {}
        kind, tail, head, r_lo, r_hi = [], [], [], [], []
        start = 0
        for k, t, h, lo, hi in blocks:
            n = t.size
            self.slices[k] = slice(start, start + n)
            start += n
            kind.append(np.full(n, k, dtype=np.int8))
            tail.append(t.ravel())
            head.append(h.ravel())
            r_lo.append(np.full(n, lo, dtype=float))
            r_hi.append(np.full(n, hi, dtype=float))
        assert start == self.num_branches

        self.kind = np.concatenate(kind)
        self.tail = np.concatenate(tail)
        self.head = np.concatenate(head)
        # Sector bounds on each branch's incremental resistance; equal for the linear branches, r_hi = inf for 1D1R cells,
        # [0, inf] for diodes on branches of their own.
        self.r_lo = np.concatenate(r_lo)
        self.r_hi = np.concatenate(r_hi)
        # Source EMF in series with each branch, nonzero only on sources: i = (v - emf) / R_source.
        self.emf = np.zeros(self.num_branches)
        self.emf[self.slices[BranchKind.SOURCE]] = self.E

    def params(self):
        """The parameters that, passed back to Config(...), regenerate this circuit."""
        as_json = lambda R: R if isinstance(R, (int, float)) else list(R)
        params = dict(num_word_lines=self.p, num_bit_lines=self.q, R_min=as_json(self.R_min),
                      R_max=as_json(self.R_max), R_wire=self.R_wire, R_source=self.R_source, R_load=self.R_load,
                      E_range=list(self.E_range), seed=self.seed, device=self.device)
        if self.device == "1s1r":
            params.update(R_on=self.R_on, R_off=self.R_off, V_th=self.V_th, delta=self.delta)
        if self.device == "1d1r_shockley":
            params.update(I_s=self.I_s, n=self.n, V_T=self.V_T)
        if self.internal_node:
            params.update(internal_node=True)
        return params

    def word_node(self, r, c):
        """Index of node w[r, c]; accepts arrays."""
        return r * self.q + c

    def bit_node(self, r, c):
        """Index of node b[r, c]; accepts arrays."""
        return self.p * self.q + r * self.q + c

    def mid_node(self, r, c):
        """Index of the internal node m[r, c] between a cell's diode and memristor (internal_node only); accepts arrays."""
        return 2 * self.p * self.q + r * self.q + c

    def line_potentials(self, v):
        """
        Potentials of the 2pq word and bit nodes (in node order) read off branch voltages v along the lines: from each
        source's voltage along its word line, and from each load's voltage up its bit line. These linear branches form a
        spanning tree of the word and bit nodes, so the potentials are exact when v satisfies KVL, and they compare
        solutions of the lumped and internal-node formulations of the same circuit, which share these nodes and branches.
        """
        p, q = self.p, self.q
        word = np.zeros((p, q))
        word[:, 0] = v[self.slices[BranchKind.SOURCE]]
        word[:, 1:] = word[:, :1] - np.cumsum(v[self.slices[BranchKind.WORD_WIRE]].reshape(p, q - 1), axis=1)
        bit = np.zeros((p, q))
        bit[-1] = v[self.slices[BranchKind.LOAD]]
        # bit wire (r, c) runs from b[r, c] to b[r + 1, c]: b[r] = b[r + 1] + v
        bit[:-1] = bit[-1] + np.cumsum(v[self.slices[BranchKind.BIT_WIRE]].reshape(p - 1, q)[::-1], axis=0)[::-1]
        return np.concatenate([word.ravel(), bit.ravel()])

    def incidence(self):
        """
        Reduced incidence matrix A (num_nodes x num_branches, ground row removed), with A[n, k] = +1 if
        node n is the tail of branch k and -1 if it is the head. KCL is A @ i = 0 and KVL is v = A.T @ u.
        """
        k = np.arange(self.num_branches)
        rows = np.concatenate([self.tail, self.head])
        cols = np.concatenate([k, k])
        vals = np.concatenate([np.ones(self.num_branches), -np.ones(self.num_branches)])
        keep = rows != GROUND
        return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])),
                             shape=(self.num_nodes, self.num_branches))

    def current(self, v, x0=None):
        """Branch currents i = g(v) for a vector of all m branch voltages. x0 optionally warm-starts the series solve of
        1S1R and lumped Shockley 1D1R cells (their two inner voltages); it changes the cost, not the result. An ideal diode
        on its own branch carries its cell's lumped current (see the class docstring)."""
        i = (v - self.emf) / self.r_hi
        d = self.slices[BranchKind.DEVICE]
        i[d] = self.device_current(v[d], x0)
        if self.diode_law == "ideal":
            e = self.slices[BranchKind.DIODE]
            i[d] = i[e] = memristor_current(np.maximum(v[d] + v[e], 0.0), self.R_min_cell, self.R_max_cell)
        elif self.diode_law == "shockley":
            e = self.slices[BranchKind.DIODE]
            i[e] = shockley_current(v[e], self.I_s, self.n * self.V_T)
        return i

    def conductance(self, v):
        """Incremental branch conductances g'(v), each within [1 / r_hi, 1 / r_lo]. At a 1D1R cell's kink (v = 0) this
        is the reverse-bias value 0, one end of the subdifferential [0, 1 / R_min]."""
        if self.diode_law == "ideal":
            raise ValueError("an ideal diode on its own branch has no finite conductance")
        g = 1 / self.r_hi
        d = self.slices[BranchKind.DEVICE]
        g[d] = self.device_conductance(v[d])
        if self.diode_law == "shockley":
            e = self.slices[BranchKind.DIODE]
            g[e] = shockley_conductance(v[e], self.I_s, self.n * self.V_T)
        return g

    def device_current(self, v, x0=None):
        """Currents of the pq cells (with internal_node, of their memristors) at cell voltages v; x0 as in current()."""
        if self.device_law == "1s1r":
            return self.series_solve(v, 0.0, x0)[1]
        if self.device_law == "1d1r_shockley":
            return self.diode_series_solve(v, 0.0, x0)[1]
        if self.device_law == "1d1r":
            v = np.maximum(v, 0.0)
        return memristor_current(v, self.R_min_cell, self.R_max_cell)

    def device_conductance(self, v):
        """Incremental conductances of the pq cells (with internal_node, of their memristors) at cell voltages v."""
        if self.device_law == "1s1r":
            _, _, v_s, v_m, _ = self.series_solve(v)
            return 1 / (1 / selector_conductance(v_s, self.R_on, self.R_off, self.V_th, self.delta)
                        + 1 / memristor_conductance(v_m, self.R_min_cell, self.R_max_cell))
        if self.device_law == "1d1r_shockley":
            _, _, v_d, v_m, _ = self.diode_series_solve(v)
            return self.diode_series_conductance(v_d, v_m)
        g = memristor_conductance(v, self.R_min_cell, self.R_max_cell)
        return np.where(v > 0, g, 0.0) if self.device_law == "1d1r" else g

    def diode_series_conductance(self, v_d, v_m):
        """Incremental conductance of lumped Shockley 1D1R cells whose diode and memristor sit at v_d and v_m: the series
        combination, 0 where the diode's conductance underflows in deep reverse bias."""
        with np.errstate(divide="ignore"):
            return 1 / (1 / shockley_conductance(v_d, self.I_s, self.n * self.V_T)
                        + 1 / memristor_conductance(v_m, self.R_min_cell, self.R_max_cell))

    def series_solve(self, a, gamma=0.0, x0=None):
        """devices.series_solve for this circuit's 1S1R cells: with gamma = 0 the cells at cell voltages a, otherwise
        their resolvent. Returns the cell voltages, currents, selector and memristor voltages, and the steps taken."""
        return series_solve(a, gamma, self.R_min_cell, self.R_max_cell, self.R_on, self.R_off, self.V_th, self.delta,
                            x0)

    def diode_series_solve(self, a, gamma=0.0, x0=None):
        """devices.diode_series_solve for this circuit's lumped Shockley 1D1R cells, as series_solve for 1S1R cells.
        Returns the cell voltages, currents, diode and memristor voltages, and the steps taken."""
        return diode_series_solve(a, gamma, self.R_min_cell, self.R_max_cell, self.I_s, self.n * self.V_T, x0)
