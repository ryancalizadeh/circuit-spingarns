from enum import IntEnum

import numpy as np
import scipy.sparse as sp

GROUND = -1


class BranchKind(IntEnum):
    """Branch kinds, in the order their blocks appear in the branch numbering."""
    WORD_WIRE = 0
    BIT_WIRE = 1
    DEVICE = 2
    SOURCE = 3
    LOAD = 4


class Config:
    """
    A configuration describing a memristive crossbar array with p word lines and q bit lines.

    Word line r carries the nodes w[r, 0..q-1] joined by wire resistances R_wire, and bit line c carries
    the nodes b[0..p-1, c] joined by the same. A memristive device joins w[r, c] to b[r, c] at every
    cross-point. Each word line is driven at its head w[r, 0] by a source E[r] with series resistance
    R_source, and each bit line is terminated at its foot b[p-1, c] by a load R_load to ground.
    Every device is a static nonlinear resistor with characteristic

        i = v / R_max + (1 / R_min - 1 / R_max) * tanh(v)

    which lies in the sector S[R_min, R_max] on incremental resistance.

    Indices are zero-based. Ground is the reference node and is not numbered, which leaves 2pq nodes:
    word nodes first, then bit nodes, each row-major over (r, c). The m = 3pq branches are numbered in
    blocks of p(q-1) word wires, q(p-1) bit wires, pq devices, p sources and q loads, each block
    row-major over (r, c); `slices` maps each BranchKind to its block.

    Branch k runs from node tail[k] to node head[k] with associated reference directions:
    v[k] = u[tail[k]] - u[head[k]] and i[k] flows from tail to head through the branch, so v[k] * i[k]
    is the power it absorbs. Devices run from w[r, c] to b[r, c]; sources and loads run to ground.

    An object of this class is used to initialize a simulation study.
    """

    def __init__(self, num_word_lines, num_bit_lines, R_min=1e3, R_max=100e3, R_wire=5.0,
                 R_source=50.0, R_load=1e3, E_range=(0.1, 0.5), seed=None):
        if num_word_lines < 1 or num_bit_lines < 1:
            raise ValueError("need at least one word line and one bit line")
        if not 0 < R_min <= R_max:
            raise ValueError("need 0 < R_min <= R_max")
        if min(R_wire, R_source, R_load) <= 0:
            raise ValueError("wire, source and load resistances must be positive")

        self.p = p = num_word_lines
        self.q = q = num_bit_lines
        self.R_min, self.R_max = R_min, R_max
        self.R_wire, self.R_source, self.R_load = R_wire, R_source, R_load
        self.E = np.random.default_rng(seed).uniform(*E_range, size=p)

        self.num_nodes = 2 * p * q
        self.num_branches = 3 * p * q

        r, c = np.indices((p, q))
        w, b = self.word_node(r, c), self.bit_node(r, c)
        blocks = [
            (BranchKind.WORD_WIRE, w[:, :-1], w[:, 1:], R_wire, R_wire),
            (BranchKind.BIT_WIRE, b[:-1, :], b[1:, :], R_wire, R_wire),
            (BranchKind.DEVICE, w, b, R_min, R_max),
            (BranchKind.SOURCE, w[:, 0], np.full(p, GROUND), R_source, R_source),
            (BranchKind.LOAD, b[-1, :], np.full(q, GROUND), R_load, R_load),
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
        # Sector bounds on each branch's incremental resistance; equal for the linear branches.
        self.r_lo = np.concatenate(r_lo)
        self.r_hi = np.concatenate(r_hi)
        # Source EMF in series with each branch, nonzero only on sources: i = (v - emf) / R_source.
        self.emf = np.zeros(self.num_branches)
        self.emf[self.slices[BranchKind.SOURCE]] = self.E

    def word_node(self, r, c):
        """Index of node w[r, c]; accepts arrays."""
        return r * self.q + c

    def bit_node(self, r, c):
        """Index of node b[r, c]; accepts arrays."""
        return self.p * self.q + r * self.q + c

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

    def current(self, v):
        """Branch currents i = g(v) for a vector of all m branch voltages."""
        i = (v - self.emf) / self.r_hi
        d = self.slices[BranchKind.DEVICE]
        i[d] += (1 / self.R_min - 1 / self.R_max) * np.tanh(v[d])
        return i

    def conductance(self, v):
        """Incremental branch conductances g'(v), each within [1 / r_hi, 1 / r_lo]."""
        g = 1 / self.r_hi
        d = self.slices[BranchKind.DEVICE]
        g[d] += (1 / self.R_min - 1 / self.R_max) * (1 - np.tanh(v[d]) ** 2)
        return g
