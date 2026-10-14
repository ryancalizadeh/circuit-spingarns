"""
The Condat-Vu primal-dual algorithm on the crossbar, as Chaffey, Banert, Giselsson and Pates apply it to circuits:
"Circuit analysis using monotone+skew splitting", European Journal of Control 74 (2023) 100854, Algorithm 1.

Element extraction (the paper's Theorem 1) splits the branches into a spanning tree, whose voltages are independent and
fix every node potential, and its cotree, the links, whose currents are independent and fix every branch current. With
A_T and A_L the columns of the incidence matrix on the tree and on the links, KCL gives the tree currents i_T = M i_L and
KVL the link voltages v_L = -M^T v_T, where M = -A_T^-1 A_L. Tree branches are written in admittance form, i = G(v), and
links in impedance form, v = R(i), each with its source EMF inside its law. A DC solution is then a zero of the sum of a
monotone and a skew operator (the paper's eq. (5)):

    0 = R(i_L) + M^T v_T,    0 = G(v_T) - M i_L,

and Algorithm 1 iterates on the link currents and the tree voltages:

    i_L <- (Id + tau R)^-1 (i_L - tau M^T v_T)
    v_T <- (Id + sigma G)^-1 (v_T + sigma M (2 i_L_new - i_L_old)),

which converges for tau sigma ||M||^2 < 1 (Condat 2013, Vu 2013). M is never formed or inverted: applying it or its
transpose is a sweep along the tree (cumulative sums along the lines for the crossbar's natural tree, a solve with the
tree's sparse incidence matrix otherwise), so an iteration costs O(m) and no large matrix is ever factorized. The price is
the step-size condition, since ||M|| grows with the lengths of the fundamental loops and with how many loops share a tree
branch: on the crossbar, in proportion to the array's side (LinesInterconnect.norm).

Resolvents, elementwise. A linear branch v = emf + R_e i: (Id + tau R)^-1(x) = (x - tau emf) / (1 + tau R_e) and
(Id + sigma G)^-1(y) = (R_e y + sigma emf) / (R_e + sigma). A memristor link: i = x - tau w, with w the root of
tau w + g_m(w) = x (devices.tanh_root), which is the link's voltage. A memristor in the tree: the root v of
v + sigma g_m(v) = y, Spingarn's device resolvent with port resistance sigma. A 1D1R cell, i = g_m(max(v, 0)), clips
the link's current at zero (i = 0 for x <= 0) and uses Spingarn's diode resolvent in the tree. In wave terms each link sees the port
resistance 1 / tau and each tree branch sigma, as in Spingarn's method, but the waves are exchanged through M explicitly,
one step late, instead of through a solve of the network.

Step sizes are set by tau sigma ||M||^2 = step_product < 1 and by rho = sqrt(sigma / tau), the geometric mean of the two
port resistances. rho = "auto" is Chambolle and Pock's choice when both operators are strongly monotone (J. Math. Imaging
Vis. 40 (2011) 120, Theorem 3): rho = sqrt(mu_R / mu_G), with mu_R the smallest incremental resistance of a link and mu_G
the smallest incremental conductance of a tree branch.
"""
import time

import numpy as np
import scipy.sparse.linalg as spla

from Config import BranchKind
from devices import tanh_root
from algorithms.common import Solver, SolveOutput, kcl_check
from algorithms.spingarns import device_resolvent

TREES = ("lines", "devices")
# Up to this many links, ||M|| is computed from the dense matrix; above, by Lanczos on M^T M. Lanczos agrees to ~1e-10 and
# is far cheaper beyond: at 16x16 (256 links) 1.8 ms against 485 ms for numpy's dense SVD on the laptop.
DENSE_NORM_LINKS = 64


def spanning_tree(cfg, rule):
    """
    The tree branches, as a boolean mask over cfg's branches; the rest are the links.

        "lines"    every wire, source and load: each word line hangs from its source and each bit line from its load,
                   and the devices are the links. On the default crossbar no tree branch (5 ohm wires, 50 ohm sources,
                   1 kohm loads) resists more than any device (at least R_min = 1 kohm), so it is a minimum spanning tree.
        "devices"  word wires, sources and devices: each bit node hangs from its word node through its device, and bit
                   wires and loads are the links.
    """
    if rule == "lines":
        return cfg.kind != BranchKind.DEVICE
    if rule == "devices":
        return np.isin(cfg.kind, (BranchKind.WORD_WIRE, BranchKind.SOURCE, BranchKind.DEVICE))
    raise ValueError(f"tree must be one of {TREES}, not {rule!r}")


class Interconnect:
    """
    The skew part of the circuit for a spanning tree: M = -A_T^-1 A_L and M^T, applied through a sparse LU of the tree's
    incidence matrix A_T. A_T's entries are 0 and +-1 and it is triangular up to ordering, so every solve is exact and the
    LU stays O(N) (SuperLU's has ~1.45x A_T's off-diagonal nonzeros on the crossbar, at every size). A set of branches
    that is not a spanning tree makes A_T singular, and the factorization raises.
    """

    def __init__(self, cfg, tree, A=None):
        A = (cfg.incidence() if A is None else A).tocsc()
        self.tree, self.links = np.flatnonzero(tree), np.flatnonzero(~tree)
        if self.tree.size != cfg.num_nodes:
            raise ValueError(f"a spanning tree has {cfg.num_nodes} branches, not {self.tree.size}")
        self.A_L = A[:, self.links].tocsr()
        self.A_LT = self.A_L.T.tocsr()
        self.lu = spla.splu(A[:, self.tree].tocsc())

    def tree_currents(self, i_L):
        """M i_L: the tree currents KCL implies for link currents i_L."""
        return -self.lu.solve(self.A_L @ i_L)

    def potentials(self, v_T):
        """The node potentials u with A_T^T u = v_T, which fix every branch voltage by KVL."""
        return self.lu.solve(v_T, trans="T")

    def link_voltages(self, u):
        """A_L^T u = -M^T v_T: the link voltages KVL implies, from the potentials of the tree voltages v_T."""
        return self.A_LT @ u

    def transpose(self, v_T):
        """M^T v_T."""
        return -self.link_voltages(self.potentials(v_T))

    def dense(self):
        """M as a dense array (num_tree x num_links); for small circuits and tests."""
        return -self.lu.solve(self.A_L.toarray())

    def norm(self):
        """||M||, the largest singular value: from the dense matrix for few links, otherwise by Lanczos on M^T M (to
        ~1e-10 relative, from below)."""
        n = self.links.size
        if n <= DENSE_NORM_LINKS:
            return float(np.linalg.norm(self.dense(), 2))
        op = spla.LinearOperator((n, n), matvec=lambda x: self.transpose(self.tree_currents(np.ravel(x))), dtype=float)
        lam = spla.eigsh(op, k=1, which="LA", tol=1e-10, ncv=24, return_eigenvectors=False)[0]
        return float(np.sqrt(lam))


class LinesInterconnect(Interconnect):
    """
    Interconnect for the "lines" tree, applied by cumulative sums along the lines instead of solves with A_T: KCL makes a
    word wire carry the current of every device beyond it on its row and a bit wire that of every device above it on its
    column, a source the negated total of its row and a load the total of its column; KVL makes a word node's potential
    its source's voltage minus the wire drops on the way, and a bit node's its load's voltage plus the drops below it.
    Same results as Interconnect to rounding, at a fraction of the cost.
    """

    def __init__(self, cfg, tree, A=None):
        super().__init__(cfg, tree, A)
        self.p, self.q = cfg.p, cfg.q
        # tree order (branch order without the devices): word wires, bit wires, sources, loads; links: the cells
        ww, bw = cfg.p * (cfg.q - 1), (cfg.p - 1) * cfg.q
        self.blocks = np.cumsum([0, ww, bw, cfg.p, cfg.q])

    def tree_currents(self, i_L):
        p, q = self.p, self.q
        I = i_L.reshape(p, q)
        beyond = np.cumsum(I[:, ::-1], axis=1)[:, ::-1]  # beyond[r, c]: cells c..q-1 of row r
        above = np.cumsum(I, axis=0)  # above[r, c]: cells 0..r of column c
        return np.concatenate([beyond[:, 1:].ravel(), above[:-1].ravel(), -beyond[:, 0], above[-1]])

    def potentials(self, v_T):
        p, q = self.p, self.q
        s = self.blocks
        v_ww, v_bw = v_T[s[0]:s[1]].reshape(p, q - 1), v_T[s[1]:s[2]].reshape(p - 1, q)
        v_src, v_load = v_T[s[2]:s[3]], v_T[s[3]:s[4]]
        u_w = np.empty((p, q))
        u_w[:, 0] = 0.0
        np.cumsum(v_ww, axis=1, out=u_w[:, 1:])
        u_w = v_src[:, None] - u_w
        u_b = np.empty((p, q))
        u_b[-1] = 0.0
        np.cumsum(v_bw[::-1], axis=0, out=u_b[-2::-1])
        u_b += v_load[None, :]
        return np.concatenate([u_w.ravel(), u_b.ravel()])

    def link_voltages(self, u):
        n = self.p * self.q
        return u[:n] - u[n:]

    def norm(self):
        return lines_norm(self.p, self.q)


def lines_norm(p, q, smallest=False):
    """
    ||M|| for the line tree of a p x q crossbar, in closed form (or, with smallest=True, M's smallest singular value).
    M^T M is the Gram matrix of the cells' fundamental loops, and two loops on one row share the source and the word wires
    up to the nearer cell, two on one column the load and the bit wires below the lower cell, both traversed the same
    way, and two others nothing. So M^T M is the Kronecker sum I_p (x) K_q + K'_p (x) I_q, where K_n = [min(j, k)] for
    j, k = 1..n and K'_p is K_p in reverse order. K_n^-1 is the second-difference matrix with a free end, whose
    eigenvalues are 4 sin^2((2k - 1) pi / (4n + 2)), k = 1..n, so ||M||^2 = 1 / (4 sin^2(pi / (4p + 2))) +
    1 / (4 sin^2(pi / (4q + 2))): about 0.90 (n + 1/2) for an n x n array; the smallest singular value (k = n) tends to
    1 / sqrt(2).
    """
    lam = lambda n: 1 / (4 * np.sin((2 * (n if smallest else 1) - 1) * np.pi / (4 * n + 2)) ** 2)
    return float(np.sqrt(lam(p) + lam(q)))


def link_resolvent(x, tau, R_min, R_max, w0, diode=False):
    """
    (Id + tau R)^-1(x) for memristor links in impedance form, R = g_m^-1, elementwise: the current i with
    i + tau g_m^-1(i) = x. With w = g_m^-1(i), the link's voltage, this is tau w + g_m(w) = x, i.e. c w + d tanh(w) = x with
    c = tau + 1 / R_max and d = 1 / R_min - 1 / R_max, solved by devices.tanh_root from the warm start w0. Returns i, w
    and the Newton steps taken.

    With diode=True the links are 1D1R cells, i = g_m(max(v, 0)). In impedance form an open cell is the half-line i = 0,
    v <= 0, so for x <= 0 the resolvent is i = 0 (w is then returned as 0, which only serves as a warm start), and for
    x > 0 it is the memristor's.
    """
    if diode:
        x = np.maximum(x, 0.0)
    w, steps = tanh_root(tau + 1 / R_max, 1 / R_min - 1 / R_max, x, w0)
    return x - tau * w, w, steps


def balanced_rho(cfg, tree):
    """Chambolle and Pock's balance of the step sizes for strongly monotone R and G: sqrt(mu_R / mu_G), with mu_R the
    smallest incremental resistance of a link and mu_G the smallest incremental conductance of a tree branch."""
    return float(np.sqrt(cfg.r_lo[~tree].min() * cfg.r_hi[tree].max()))


class CondatVuSolver(Solver):
    """
    Algorithm 1 of Chaffey et al. (2023), the Condat-Vu iteration on the monotone + skew form of the circuit (see the
    module docstring), with scalar step sizes as in the paper.

    tree picks the spanning tree (see spanning_tree). step_product = tau sigma ||M||^2 must lie in (0, 1); rho is
    sqrt(sigma / tau) in ohms, or "auto" for Chambolle and Pock's balance (balanced_rho). abstol is the absolute current
    tolerance of the KCL test, as in SPICE.

    Convergence is judged as for the other solvers. The tree voltages fix the node potentials u, so KVL holds at every
    iterate; the test is SPICE's own KCL test (kcl_check) on the branch currents that the exact branch laws give at u. The
    potentials are needed for the next link update anyway, so the test costs no extra solve. The output is those branch
    voltages and currents. Iterations count passes of the loop, i.e. updates of both the link currents and the tree
    voltages.
    """
    name = "condat_vu"
    default_max_iterations = 100_000

    def __init__(self, tol=None, max_iterations=None, rho="auto", step_product=0.99, tree="lines", abstol=1e-12):
        if not (rho == "auto" or isinstance(rho, (int, float)) and not isinstance(rho, bool) and rho > 0):
            raise ValueError(f"rho must be 'auto' or a positive resistance, not {rho!r}")
        if not (isinstance(step_product, (int, float)) and 0 < step_product < 1):
            raise ValueError(f"step_product must lie in (0, 1), not {step_product!r}")
        if tree not in TREES:
            raise ValueError(f"tree must be one of {TREES}, not {tree!r}")
        super().__init__(tol, max_iterations, rho=rho, step_product=step_product, tree=tree, abstol=abstol)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def solve(self, cfg):
        if cfg.device not in ("memristor", "1d1r"):
            raise ValueError(f"condat_vu handles memristor and 1d1r cells only, not {cfg.device!r}")
        diode = cfg.device == "1d1r"
        abstol = self.options["abstol"]
        history = {}
        t = time.perf_counter()
        A = cfg.incidence()
        tree = spanning_tree(cfg, self.options["tree"])
        net = (LinesInterconnect if self.options["tree"] == "lines" else Interconnect)(cfg, tree, A)
        t_norm = time.perf_counter()
        norm_M = net.norm()
        history["norm_time"] = time.perf_counter() - t_norm
        rho = balanced_rho(cfg, tree) if self.options["rho"] == "auto" else float(self.options["rho"])
        if not np.isfinite(rho):  # 1D1R cells in the tree: an open cell conducts nothing, so mu_G = 0
            raise ValueError("no automatic step balance for a tree holding 1d1r cells; give rho in ohms")
        root = np.sqrt(self.options["step_product"])
        tau, sigma = root / (rho * norm_M), rho * root / norm_M
        history.update(setup_time=time.perf_counter() - t, norm_M=norm_M, rho=rho, tau=tau, sigma=sigma,
                       num_links=int(net.links.size))

        # element data on each side: device cells (their index among the pq cells) and linear branches
        d = cfg.slices[BranchKind.DEVICE]
        L, T = net.links, net.tree
        L_dev, T_dev = cfg.kind[L] == BranchKind.DEVICE, cfg.kind[T] == BranchKind.DEVICE
        L_cell, T_cell = L[L_dev] - d.start, T[T_dev] - d.start
        L_lin, T_lin = ~L_dev, ~T_dev
        R_L, emf_L = cfg.r_hi[L][L_lin], cfg.emf[L][L_lin]
        R_T, emf_T = cfg.r_hi[T][T_lin], cfg.emf[T][T_lin]
        Rn_L, Rx_L = cfg.R_min_cell[L_cell], cfg.R_max_cell[L_cell]
        Rn_T, Rx_T = cfg.R_min_cell[T_cell], cfg.R_max_cell[T_cell]
        w_L, w_T = np.zeros(L_cell.size), np.zeros(T_cell.size)  # warm starts: the devices' voltages

        i_L, v_T = np.zeros(L.size), np.zeros(T.size)
        u = np.zeros(cfg.num_nodes)  # potentials of v_T
        i_new, v_new = np.empty(L.size), np.empty(T.size)
        scale_i, scale_v = 1 / tau, 1 / sigma  # the iteration's own metric, in which it is nonexpansive
        kcl, fixed_point, newton_steps = [], [], 0
        converged = False
        t = time.perf_counter()
        for k in range(self.max_iterations):
            # 1) links: i <- (Id + tau R)^-1 (i - tau M^T v), where -M^T v = A_L^T u
            x = i_L + tau * net.link_voltages(u)
            i_new[L_dev], w_L, steps = link_resolvent(x[L_dev], tau, Rn_L, Rx_L, w_L, diode)
            newton_steps += steps
            i_new[L_lin] = (x[L_lin] - tau * emf_L) / (1 + tau * R_L)

            # 2) tree: v <- (Id + sigma G)^-1 (v + sigma M (2 i_new - i))
            y = v_T + sigma * net.tree_currents(2 * i_new - i_L)
            v_new[T_lin] = (R_T * y[T_lin] + sigma * emf_T) / (R_T + sigma)
            if T_cell.size:
                w_T, steps = device_resolvent(y[T_dev], sigma, Rn_T, Rx_T, w_T, diode=diode)
                v_new[T_dev] = w_T
                newton_steps += steps

            di, dv = i_new - i_L, v_new - v_T
            change = scale_i * (di @ di) + scale_v * (dv @ dv)
            size = scale_i * (i_new @ i_new) + scale_v * (v_new @ v_new)
            fixed_point.append(float(np.sqrt(change / max(size, 1e-300))))
            i_L, i_new = i_new, i_L
            v_T, v_new = v_new, v_T

            # KCL at the potentials of the tree voltages, with every branch on its exact law
            u = net.potentials(v_T)
            v, i, kcl_ok, kcl_abs, kcl_rel = kcl_check(cfg, u, self.tol, abstol, A)
            kcl.append(kcl_rel)
            if kcl_ok:
                converged = True
                break
        history["iteration_time"] = time.perf_counter() - t
        history.update(kcl_residual=kcl_abs, kcl_residual_rel=kcl_rel, newton_steps=newton_steps, kcl_history=kcl,
                       fixed_point_history=fixed_point)
        return SolveOutput(v, i, converged, k + 1, history)
