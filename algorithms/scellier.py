"""
Exact coordinate descent on the crossbar's energy, after B. Scellier, "A Fast Algorithm to Simulate Nonlinear Resistive
Networks", ICML 2024 (arXiv 2402.11674).

Every branch here is a monotone two-terminal element i_e = g_e(v_e), so it has a convex content Phi_e (the integral of
g_e), and the DC solution minimizes the energy E(u) = sum_e Phi_e((A^T u)_e) over the node potentials u: KCL,
A g(A^T u) = 0, is grad E = 0. For linear resistors Phi is half the power dissipated, which is the paper's Theorem 1; for
the tanh memristor and the 1D1R and 1S1R cells it is Millar's (1951) generalization, which the paper cites. Exact
coordinate descent minimizes E over one node potential at a time, the others fixed: it solves that node's KCL equation
alone, i.e. nonlinear Gauss-Seidel. The paper's Theorem 2 is this update for ideal resistors, diodes and sources.

The paper's ideal diode is a constraint between two node potentials. Between two free nodes (a 1D1R cell split into a
diode and a resistor with a node between them) single-node descent can stop short of the minimum: from u = 0 neither end
of the diode may move alone. As in the rest of the repo the diode stays inside the cell's law, i = g_m(max(v, 0)), whose
content is differentiable, so descent converges.

Node update. Every node of the crossbar touches exactly one cell, plus linear branches of total conductance G: its wires
and, at the end of a line, a source or a load. With P = (sum of g_j u_j) / G over those neighbours (a source's far node at
E, ground at 0), a word node's KCL is x + g_cell(x) / G = P - u_b for x = u_w - u_b, and a bit node's is
x + g_cell(x) / G = u_w - P. Both are the cell's resolvent with port resistance 1 / G, so the update reuses the cells'
code from the Spingarn solver; for a linear cell it is the paper's conductance-weighted average.

Blocks. The crossbar's node graph is bipartite: colour word node (r, c) by (r + c) mod 2 and bit node (r, c) by
(r + c + 1) mod 2, and every wire and every cell joins the two colours (sources and loads end at fixed nodes). All nodes of
one colour can then be updated at once from the other colour's values, which is the paper's even/odd-layer block update for
deep resistive networks ("nodes"). But each line is a chain of low-resistance wires, along which such updates carry
information about one node per sweep. "lines" applies the paper's idea one level up: all word nodes, then all bit nodes.
Word lines do not touch each other, so with the bit lines fixed each word line is a convex problem of its own in its q
potentials, with a tridiagonal Hessian; Newton's method solves it exactly. Likewise each bit line.

An iteration is one sweep: both colours, or both line blocks. Convergence is judged as for the other solvers. Every
iterate is a set of node potentials, so KVL holds and the branch laws give the currents, and the test is SPICE's own KCL
test (kcl_check), every check_every sweeps. The paper instead runs a fixed number of sweeps (4-6) and reports no residual.
"""
import time

import numpy as np
from scipy.linalg import solve_banded

from devices import series_solve
from algorithms.common import Solver, SolveOutput, kcl_check
from algorithms.spingarns import device_resolvent

BLOCKS = ("nodes", "lines")
# A line block counts as solved when every node's KCL holds to this fraction of the currents it sums.
LINE_RTOL = 1e-13
LINE_MAX_NEWTON = 50
KINK = 1e-12  # a cell crosses the 1D1R kink when its voltage moves across this many volts


def tridiagonal_solve(diag, off, rhs):
    """Solve the symmetric tridiagonal system with diagonal diag and off-diagonal off (off[k] couples rows k, k + 1)."""
    n = diag.size
    ab = np.zeros((3, n))
    ab[0, 1:], ab[1], ab[2, :-1] = off, diag, off
    return solve_banded((1, 1), ab, rhs, overwrite_ab=True, check_finite=False)


class CrossbarState:
    """
    Node potentials of a crossbar as p x q arrays W (word nodes) and B (bit nodes), starting at zero, with the updates of
    exact coordinate descent on them (see the module docstring).
    """

    def __init__(self, cfg):
        if cfg.device not in ("memristor", "1d1r", "1s1r") or cfg.internal_node:
            raise ValueError(f"scellier handles lumped memristor, 1d1r and 1s1r cells only, not {cfg.device!r}"
                             + (" with internal nodes" if cfg.internal_node else ""))
        self.cfg = cfg
        p, q = cfg.p, cfg.q
        r, c = np.indices((p, q))
        self.g_wire = g = 1 / cfg.R_wire
        # each node's linear conductance G, and the current its fixed neighbour drives into it through that conductance
        self.G_w = g * ((c > 0).astype(float) + (c < q - 1)) + (c == 0) / cfg.R_source
        self.S_w = np.where(c == 0, cfg.E[:, None] / cfg.R_source, 0.0)
        self.G_b = g * ((r > 0).astype(float) + (r < p - 1)) + (r == p - 1) / cfg.R_load
        self.W, self.B = np.zeros((p, q)), np.zeros((p, q))
        # warm starts of the cell solves: cell voltages, or for 1S1R cells the selector and memristor voltages
        self.x = np.zeros(p * q)
        self.v_s, self.v_m = np.zeros(p * q), np.zeros(p * q)
        # node blocks: word nodes of colour k, whose cells' bit nodes have the other colour, and bit nodes of colour k
        self.word_colour = [((r + c) % 2 == k) for k in (0, 1)]
        # line blocks: wires couple neighbours along each line; lines are stacked row-major (word) and column-major (bit)
        self.off_w = np.where(c < q - 1, -g, 0.0).ravel()[:-1]
        self.off_b = np.where(r < p - 1, -g, 0.0).T.ravel()[:-1]

    def potentials(self):
        """All node potentials in Config's node order: word nodes, then bit nodes, each row-major."""
        return np.concatenate([self.W.ravel(), self.B.ravel()])

    def warm_1s1r(self):
        return (self.v_s, self.v_m) if self.cfg.device == "1s1r" else None

    def _word_sum(self, W):
        """sum of g u_j over each word node's wire neighbours"""
        s = np.zeros_like(W)
        s[:, 1:] += W[:, :-1]
        s[:, :-1] += W[:, 1:]
        return self.g_wire * s

    def _bit_sum(self, B):
        s = np.zeros_like(B)
        s[1:] += B[:-1]
        s[:-1] += B[1:]
        return self.g_wire * s

    def _cells(self, mask, a, gamma):
        """The resolvent x + gamma g_cell(x) = a of the cells under mask (a p x q boolean array); returns x and steps."""
        cfg, m = self.cfg, mask.ravel()
        if cfg.device == "1s1r":
            x, _, v_s, v_m, steps = series_solve(a, gamma, cfg.R_min_cell[m], cfg.R_max_cell[m], cfg.R_on, cfg.R_off,
                                                 cfg.V_th, cfg.delta, (self.v_s[m], self.v_m[m]))
            self.v_s[m], self.v_m[m] = v_s, v_m
        else:
            x, steps = device_resolvent(a, gamma, cfg.R_min_cell[m], cfg.R_max_cell[m], self.x[m],
                                        diode=cfg.device == "1d1r")
        self.x[m] = x
        return x, steps

    def node_sweep(self):
        """One sweep of single-node updates, one colour at a time. Returns the cell-solve steps taken."""
        steps = 0
        for w in self.word_colour:
            b = ~w  # bit nodes of the same colour as the word nodes under w
            P_w = (self._word_sum(self.W) + self.S_w) / self.G_w
            P_b = self._bit_sum(self.B) / self.G_b
            # each word node of this colour against its cell's bit node, and each bit node against its word node: all of
            # the other colour, so the two updates are independent
            x_w, s_w = self._cells(w, P_w[w] - self.B[w], 1 / self.G_w[w])
            x_b, s_b = self._cells(b, self.W[b] - P_b[b], 1 / self.G_b[b])
            self.W[w] = self.B[w] + x_w
            self.B[b] = self.W[b] - x_b
            steps += s_w + s_b
        return steps

    def _block_residual(self, word):
        """Net current leaving every word node (word=True) or bit node, as a p x q array, and the sum of the magnitudes
        of the currents it adds up."""
        g_cell = self.cfg.device_current((self.W - self.B).ravel()).reshape(self.W.shape)
        if word:
            F = self.G_w * self.W - self._word_sum(self.W) - self.S_w + g_cell
            scale = self.G_w * np.abs(self.W) + self._word_sum(np.abs(self.W)) + np.abs(self.S_w)
        else:
            F = self.G_b * self.B - self._bit_sum(self.B) - g_cell
            scale = self.G_b * np.abs(self.B) + self._bit_sum(np.abs(self.B))
        return F, scale + np.abs(g_cell)

    def _line_block(self, word):
        """
        Minimize exactly over every word line (word=True) or every bit line, the other block fixed, by Newton's method
        with the lines' tridiagonal Hessians. Returns the Newton steps taken.

        At a 1D1R cell's kink the full Newton step can overshoot and cycle (seen at 128x128: two states alternating for
        50 steps). So on each line where a step moves a cell across the kink, the step is halved until the line's energy
        is still falling at its end (phi'(t) = -d . F(u - t d) <= 0), which holds for small enough t since the energy is
        convex and differentiable. Steps that cross no kink, which include every step near the solution, are taken whole.
        """
        cfg, p, q = self.cfg, self.cfg.p, self.cfg.q
        U, axis = (self.W, 1) if word else (self.B, 0)  # U is updated in place; axis runs along the lines
        per_line = (lambda t: t[:, None]) if word else (lambda t: t[None, :])
        for step in range(LINE_MAX_NEWTON + 1):
            F, scale = self._block_residual(word)
            if np.all(np.abs(F) <= LINE_RTOL * scale):
                return step
            if step == LINE_MAX_NEWTON:
                break
            x = self.W - self.B
            dg = cfg.device_conductance(x.ravel()).reshape(p, q)
            if word:
                d = tridiagonal_solve((self.G_w + dg).ravel(), self.off_w, F.ravel()).reshape(p, q)
            else:
                d = tridiagonal_solve((self.G_b + dg).T.ravel(), self.off_b, F.T.ravel()).reshape(q, p).T
            U -= d
            if cfg.device != "1d1r":
                continue
            check = ((x > KINK) != (self.W - self.B > KINK)).any(axis=axis)
            t = np.ones(check.size)
            for _ in range(60):
                if not check.any():
                    break
                F_t, _ = self._block_residual(word)
                check &= (d * F_t).sum(axis=axis) < 0  # phi'(t) > 0: the step overshoots the line's minimum along d
                U += per_line(np.where(check, t / 2, 0.0)) * d
                t[check] /= 2
        raise RuntimeError(f"Newton's method did not solve the {'word' if word else 'bit'} lines")

    def line_sweep(self):
        """One sweep of line blocks: every word line, then every bit line. Returns the Newton steps taken."""
        return self._line_block(word=True) + self._line_block(word=False)


class ScellierSolver(Solver):
    """
    Exact coordinate descent on the crossbar's energy (see the module docstring). blocks is "nodes" (one colour of the
    checkerboard at a time, the paper's even/odd update) or "lines" (all word lines, then all bit lines). check_every
    sets how many sweeps pass between KCL tests; abstol is the test's absolute current tolerance, as in SPICE.

    Starts from u = 0. Iterations count sweeps; with check_every > 1 the count is a multiple of it.
    """
    name = "scellier"
    default_blocks = "nodes"
    default_max_iterations = 5_000_000

    def __init__(self, tol=None, max_iterations=None, blocks=None, check_every=1, abstol=1e-12):
        blocks = self.default_blocks if blocks is None else blocks
        if blocks not in BLOCKS:
            raise ValueError(f"blocks must be one of {BLOCKS}, not {blocks!r}")
        if not (type(check_every) is int and check_every >= 1):
            raise ValueError(f"check_every must be a positive integer, not {check_every!r}")
        super().__init__(tol, max_iterations, blocks=blocks, check_every=check_every, abstol=abstol)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def solve(self, cfg):
        check_every, abstol = self.options["check_every"], self.options["abstol"]
        t = time.perf_counter()
        A = cfg.incidence()
        state = CrossbarState(cfg)
        sweep = state.node_sweep if self.options["blocks"] == "nodes" else state.line_sweep
        history = dict(setup_time=time.perf_counter() - t)
        kcl, steps, converged = [], 0, False
        t = time.perf_counter()
        for k in range(1, self.max_iterations + 1):
            steps += sweep()
            if k % check_every and k < self.max_iterations:
                continue
            v, i, ok, kcl_abs, kcl_rel = kcl_check(cfg, state.potentials(), self.tol, abstol, A, state.warm_1s1r())
            kcl.append(kcl_rel)
            if ok:
                converged = True
                break
        history.update(iteration_time=time.perf_counter() - t, kcl_residual=kcl_abs, kcl_residual_rel=kcl_rel,
                       inner_steps=steps, kcl_history=kcl)
        return SolveOutput(v, i, converged, k, history)


class ScellierLinesSolver(ScellierSolver):
    """ScellierSolver with line blocks: every word line, then every bit line, each minimized exactly."""
    name = "scellier_lines"
    default_blocks = "lines"
    default_max_iterations = 1_000_000
