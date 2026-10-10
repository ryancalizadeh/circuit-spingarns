"""
Spingarn's method of partial inverses on the crossbar, run in wave variables.

A DC solution is a pair of branch voltages v in range(A^T) (KVL) and branch currents i in ker(A) (KCL) with
i_e = g_e(v_e) on every branch. Given a diagonal matrix Gamma of positive port resistances, branch e carries the incident
wave a_e = v_e + gamma_e i_e and the reflected wave b_e = v_e - gamma_e i_e. Starting from a = 0, each iteration scatters
the waves at the branches and then at the network:

    1) v = J_{Gamma T}(a), branch by branch, and b = 2 v - a;
    2) a <- (1 - alpha) a + alpha S b,  with S = 2 M - I.

J_{gamma_e T_e} = (I + gamma_e T_e)^{-1} solves v + gamma_e g_e(v) = a. M = A^T (A Gamma^{-1} A^T)^{-1} A Gamma^{-1}
projects onto range(A^T) along Gamma ker(A), orthogonally in the inner product weighted by Gamma^{-1}, so S maps reflected
waves to incident waves that satisfy KVL and KCL. A Gamma^{-1} A^T is the nodal conductance matrix of the crossbar with
every branch replaced by its port resistance; it is factorized once, so an iteration costs one pair of triangular solves.
Each iterate gives v and i = (a - b) / (2 Gamma), which satisfy the branch laws exactly; KVL and KCL hold at the fixed
point.

The matrix is symmetric positive definite. By default it is factorized by CHOLMOD's supernodal Cholesky (through cvxopt)
in a geometric nested-dissection ordering read off the crossbar's grid (algorithms/ordering.py); SuperLU's LU with
minimum-degree ordering, the original implementation, remains selectable, and either library can also use its own
fill-reducing ordering instead.

In the scaled waves Gamma^{-1/2} a, S is an orthogonal reflection, and the branch reflection a_e -> b_e is Lipschitz with
constant max |R - gamma_e| / (R + gamma_e) over branch e's incremental resistances R. With alpha = 1 the iteration is
therefore a contraction by the worst such factor, whatever the size or topology of the array. Impedance matching chooses
Gamma to make the factors small: gamma_e = R absorbs a linear branch's wave completely.
"""
import ctypes
import functools
import time
from contextlib import contextmanager
from pathlib import Path

import cvxopt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cvxopt import cholmod

from Config import BranchKind
from devices import tanh_root
from algorithms.common import Solver, SolveOutput, kcl_check
from algorithms.ordering import nested_dissection

GAMMA_RULES = ("geometric", "r_lo")
FACTORIZATIONS = ("cholmod", "slu")
# "builtin" leaves the ordering to the factorization library: minimum degree on A + A^T for SuperLU, and CHOLMOD's default
# strategy, which is AMD in cvxopt's build (compiled without METIS, so CHOLMOD has no nested dissection of its own).
ORDERINGS = ("nested_dissection", "builtin")


def port_resistances(cfg, rule):
    """
    Gamma's diagonal, one port resistance per branch. Linear branches get their own resistance. Each device, whose
    incremental resistance ranges over its sector [r_lo, r_hi] (that cell's own values), gets by `rule`:

        "geometric"  sqrt(r_lo r_hi), which minimizes the worst reflection factor over the whole sector;
        "r_lo"       r_lo: for a memristor or 1D1R cell R_min, the memristor's incremental resistance at zero bias;
                     for a 1S1R cell R_on + R_min, its on state;
        a number     that resistance in ohms, for every cell.

    A 1D1R cell's sector is unbounded (r_hi = inf), so "geometric" uses its memristor's range [R_min, R_max]. Its rule
    only matters while the diode conducts: reverse-biased, the cell is open and reflects its wave unchanged (factor +1)
    whatever its port resistance.
    """
    gamma = cfg.r_hi.copy()  # r_lo = r_hi = R on linear branches
    d = cfg.slices[BranchKind.DEVICE]
    if rule == "geometric":
        gamma[d] = np.sqrt(cfg.r_lo[d] * np.where(np.isinf(cfg.r_hi[d]), cfg.R_max_cell, cfg.r_hi[d]))
    elif rule == "r_lo":
        gamma[d] = cfg.r_lo[d]
    else:
        gamma[d] = float(rule)
    return gamma


def reflection(R, gamma):
    """Reflection factor (R - gamma) / (R + gamma) of a branch of incremental resistance R at port resistance gamma;
    +1 for R = inf (an open circuit)."""
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(R), 1.0, (R - gamma) / (R + gamma))


def contraction_bound(cfg, gamma, alpha):
    """A priori bound on the factor by which each iteration shrinks the distance to the fixed point (see above). It is
    1 for 1D1R cells, whose reverse-biased diodes reflect fully: no rate is guaranteed then, only convergence, and only
    for alpha < 1."""
    rho = np.maximum(np.abs(reflection(cfg.r_lo, gamma)), np.abs(reflection(cfg.r_hi, gamma)))
    return float(1 - alpha + alpha * rho.max())


def device_resolvent(a, gamma, R_min, R_max, v0, diode=False):
    """
    J_{gamma T}(a) for memristors, elementwise: the root v of c v + d tanh(v) = a with c = 1 + gamma / R_max and
    d = gamma (1 / R_min - 1 / R_max), by Newton's method from the warm start v0 (devices.tanh_root). Returns v and the
    number of steps. gamma, R_min and R_max may be scalars or per-device arrays.

    With diode=True the device is a 1D1R cell, i = g_m(max(v, 0)), and the equation is v + gamma g_m(max(v, 0)) = a: for
    a <= 0 the cell is open and v = a; for a > 0 the root is the memristor's.
    """
    c = 1 + gamma / R_max
    d = gamma * (1 / R_min - 1 / R_max)
    # reverse-biased 1D1R cells solve for v = 0 at once and are overwritten below
    v, steps = tanh_root(c, d, np.maximum(a, 0.0) if diode else a, v0)
    return (np.where(a > 0, v, a) if diode else v), steps


def cell_resolvent(cfg, a, gamma, warm):
    """
    J_{gamma T}(a) on cfg's pq cells, whatever their device: the cell voltages, the steps taken, and the warm start for
    the next call. warm is the previous cell voltages, or for 1S1R cells the previous (selector, memristor) voltages or
    None; a 1S1R cell's resolvent is devices.series_solve.
    """
    if cfg.device == "1s1r":
        v, _, v_s, v_m, steps = cfg.series_solve(a, gamma, warm)
        return v, steps, (v_s, v_m)
    v, steps = device_resolvent(a, gamma, cfg.R_min_cell, cfg.R_max_cell, warm, diode=cfg.device == "1d1r")
    return v, steps, v


class _CholmodFactor(ctypes.Structure):
    """The cholmod_factor struct of CHOLMOD 5 (cholmod.h), whose statistics cvxopt does not expose."""
    _fields_ = [("n", ctypes.c_size_t), ("minor", ctypes.c_size_t),
                ("Perm", ctypes.c_void_p), ("ColCount", ctypes.c_void_p), ("IPerm", ctypes.c_void_p),
                ("nzmax", ctypes.c_size_t), ("p", ctypes.c_void_p), ("i", ctypes.c_void_p), ("x", ctypes.c_void_p),
                ("z", ctypes.c_void_p), ("nz", ctypes.c_void_p), ("next", ctypes.c_void_p), ("prev", ctypes.c_void_p),
                ("nsuper", ctypes.c_size_t), ("ssize", ctypes.c_size_t), ("xsize", ctypes.c_size_t),
                ("maxcsize", ctypes.c_size_t), ("maxesize", ctypes.c_size_t),
                ("super", ctypes.c_void_p), ("pi", ctypes.c_void_p), ("px", ctypes.c_void_p), ("s", ctypes.c_void_p),
                ("ordering", ctypes.c_int), ("is_ll", ctypes.c_int), ("is_super", ctypes.c_int),
                ("is_monotonic", ctypes.c_int), ("itype", ctypes.c_int), ("xtype", ctypes.c_int),
                ("dtype", ctypes.c_int), ("useGPU", ctypes.c_int)]


_CHOLMOD_ORDERINGS = ("natural", "given", "amd", "metis", "nesdis", "colamd", "postordered")
_CHOLMOD_LONG, _CHOLMOD_REAL, _CHOLMOD_DOUBLE = 2, 1, 0
_capsule_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi))


def cholmod_factor_stats(F, n):
    """
    What CHOLMOD reports of the factor F of an n x n matrix: the ordering it used, by CHOLMOD's name ("amd" for its own
    choice here, "given" for a permutation passed to it), the entries it stores for L (factor_nnz, supernodal padding
    included) and the exact number of nonzeros in L (lnz). Read from the cholmod_factor struct inside cvxopt's factor
    object; an empty dict if the struct does not look like a double-precision factor of order n, e.g. under another
    CHOLMOD version's layout.
    """
    try:
        f = _CholmodFactor.from_address(_capsule_pointer(F, b"CHOLMOD FACTOR D L"))
    except ValueError:
        return {}
    if not (f.n == n and f.itype == _CHOLMOD_LONG and f.xtype == _CHOLMOD_REAL and f.dtype == _CHOLMOD_DOUBLE
            and f.is_super in (0, 1) and 0 <= f.ordering < len(_CHOLMOD_ORDERINGS) and f.ColCount):
        return {}
    column_counts = np.ctypeslib.as_array((ctypes.c_int64 * n).from_address(f.ColCount))
    return dict(cholmod_ordering=_CHOLMOD_ORDERINGS[f.ordering], factor_nnz=int(f.xsize if f.is_super else f.nzmax),
                lnz=int(column_counts.sum()))


@functools.cache
def _openblas():
    """The OpenBLAS library bundled with cvxopt, which CHOLMOD calls, or None if it cannot be found."""
    package = Path(cvxopt.__file__).parent
    for folder in (package / ".libs", package.parent / "cvxopt.libs"):
        for path in sorted(folder.glob("*openblas*")):
            try:
                lib = ctypes.CDLL(str(path))
                lib.openblas_get_num_threads, lib.openblas_set_num_threads
            except (OSError, AttributeError):
                continue
            return lib
    return None


@contextmanager
def blas_threads(n):
    """
    Run the block with CHOLMOD's BLAS limited to n threads (None: OpenBLAS's own setting) and restore the previous
    setting afterwards. Yields the number of threads in effect, or None if cvxopt's OpenBLAS cannot be reached, in which
    case the setting is left alone.
    """
    lib = _openblas()
    if lib is None:
        yield None
        return
    before = lib.openblas_get_num_threads()
    if n is not None:
        lib.openblas_set_num_threads(n)
    try:
        yield lib.openblas_get_num_threads()
    finally:
        lib.openblas_set_num_threads(before)


def factorize(K, factorization, perm=None):
    """
    Factorize the sparse symmetric positive definite matrix K with `factorization`, "cholmod" (supernodal Cholesky
    L L^T) or "slu" (SuperLU's L U, pivoting on the diagonal), eliminating its rows in the order perm: perm[k] is the k-th
    row, and None lets the library choose (minimum degree for SuperLU, CHOLMOD's default strategy).

    Returns solve, which maps a right-hand side to the solution of K x = rhs, and a dict describing the factorization:
    ordering_used ("mmd", "nested_dissection" or CHOLMOD's own choice) and factor_nnz, the entries the factorization
    stores (L and U for SuperLU, L with its supernodal padding for CHOLMOD); for CHOLMOD also cholmod_ordering and lnz
    (see cholmod_factor_stats).
    """
    n = K.shape[0]
    if factorization == "slu":
        # Without pivoting, which is safe for an SPD matrix, rows are eliminated in the same order as columns.
        options = dict(diag_pivot_thresh=0, options=dict(SymmetricMode=True))
        if perm is None:
            lu = spla.splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A", **options)
            return lu.solve, dict(ordering_used="mmd", factor_nnz=int(lu.nnz))
        lu = spla.splu(K[perm][:, perm].tocsc(), permc_spec="NATURAL", **options)
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(n)
        return (lambda rhs: lu.solve(rhs[perm])[inverse]), dict(ordering_used="nested_dissection",
                                                                 factor_nnz=int(lu.nnz))

    lower = sp.tril(K, format="coo")
    Kc = cvxopt.spmatrix(cvxopt.matrix(lower.data), cvxopt.matrix(lower.row.astype(np.int64)),
                         cvxopt.matrix(lower.col.astype(np.int64)), (n, n))
    # nmethods = 1 makes CHOLMOD use perm as given, 0 selects its default strategy. cvxopt treats p=None as a
    # permutation, so p is passed only when there is one.
    saved = dict(cholmod.options)
    cholmod.options["nmethods"] = 0 if perm is None else 1
    try:
        F = cholmod.symbolic(Kc, uplo="L", **({} if perm is None else {"p": cvxopt.matrix(perm)}))
        cholmod.numeric(Kc, F)
    finally:
        cholmod.options.clear()
        cholmod.options.update(saved)
    info = dict(ordering_used=None, factor_nnz=None)
    info.update(cholmod_factor_stats(F, n))
    info["ordering_used"] = "nested_dissection" if perm is not None else info.get("cholmod_ordering", "cholmod_default")
    B = cvxopt.matrix(0.0, (n, 1))
    b = np.asarray(B)[:, 0]  # a view onto B, which cholmod.solve overwrites with the solution

    def solve(rhs):
        b[:] = rhs
        cholmod.solve(F, B)
        return b.copy()

    return solve, info


class SpingarnSolver(Solver):
    """
    Spingarn's method of partial inverses with impedance-matching preconditioning (see the module docstring).

    gamma picks the devices' port resistance (see port_resistances); None selects the class's default_gamma. This class
    matches the devices at zero bias, where the crossbar's devices mostly operate; SpingarnSectorSolver matches them over
    their whole sector. alpha in (0, 1] is the relaxation: 1/2 is Spingarn's original method, and 1 contracts fastest
    when every branch is strictly increasing, as here. abstol is the absolute current tolerance of the KCL test, as in
    SPICE.

    factorization ("cholmod" or "slu") and ordering ("nested_dissection" or "builtin") choose how A Gamma^-1 A^T is
    factorized (see factorize); None selects the class's defaults, CHOLMOD in the geometric nested-dissection ordering
    here. blas_threads caps the threads of CHOLMOD's BLAS (None: OpenBLAS's default, one per core). The default of 1 is
    measured: on the 8-core laptop of 2026-10-08, 1 thread factorized as fast as 2 and 2-10x faster than all 8, whose
    threads mostly wait on each other over the many small supernodes. SuperLU is single-threaded and unaffected.

    The iterate (v, i) satisfies the branch laws exactly, so convergence is judged on the constraints it meets only in
    the limit, both to relative tolerance tol: KCL by SPICE's own test (kcl_check) on the node potentials u of v, which
    the projection yields at no extra solve, and KVL by max |v - A^T u| <= tol * max |v|. A converged answer thus passes
    the same KCL test as SPICE's. Iterations count passes of the loop, i.e. linear solves.
    """
    name = "spingarn"
    default_gamma = "r_lo"
    default_factorization = "cholmod"
    default_ordering = "nested_dissection"

    def __init__(self, tol=None, max_iterations=None, gamma=None, alpha=1.0, abstol=1e-12, factorization=None,
                 ordering=None, blas_threads=1):
        gamma = self.default_gamma if gamma is None else gamma
        factorization = self.default_factorization if factorization is None else factorization
        ordering = self.default_ordering if ordering is None else ordering
        if not (gamma in GAMMA_RULES if isinstance(gamma, str) else isinstance(gamma, (int, float)) and gamma > 0):
            raise ValueError(f"gamma must be one of {GAMMA_RULES} or a positive resistance, not {gamma!r}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must lie in (0, 1], not {alpha!r}")
        if factorization not in FACTORIZATIONS:
            raise ValueError(f"factorization must be one of {FACTORIZATIONS}, not {factorization!r}")
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, not {ordering!r}")
        if not (blas_threads is None or type(blas_threads) is int and blas_threads >= 1):
            raise ValueError(f"blas_threads must be a positive integer or None, not {blas_threads!r}")
        super().__init__(tol, max_iterations, gamma=gamma, alpha=alpha, abstol=abstol, factorization=factorization,
                         ordering=ordering, blas_threads=blas_threads)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def solve(self, cfg):
        with blas_threads(self.options["blas_threads"]) as threads:
            return self._solve(cfg, threads if self.options["factorization"] == "cholmod" else None)

    def _solve(self, cfg, threads):
        alpha, abstol = self.options["alpha"], self.options["abstol"]
        history = {}
        t = time.perf_counter()
        A = cfg.incidence()
        AT = A.T.tocsr()
        gamma = port_resistances(cfg, self.options["gamma"])
        K = A @ sp.diags(1 / gamma) @ AT
        perm, ordering_time = None, 0.0
        if self.options["ordering"] == "nested_dissection":
            t_order = time.perf_counter()
            perm = nested_dissection(cfg)
            ordering_time = time.perf_counter() - t_order
        t_factor = time.perf_counter()
        solve_K, factor_info = factorize(K, self.options["factorization"], perm)
        done = time.perf_counter()
        history.update(setup_time=done - t, ordering_time=ordering_time, factor_time=done - t_factor,
                       blas_threads=threads, **factor_info)
        history["contraction_bound"] = contraction_bound(cfg, gamma, alpha)

        d = cfg.slices[BranchKind.DEVICE]
        # J_{gamma T}(a) = lin_gain * a + lin_offset on linear branches; written to stay finite where r_hi = inf (1D1R
        # cells, whose entries are overwritten by the device resolvent anyway)
        lin_gain = 1 / (1 + gamma / cfg.r_hi)
        lin_offset = gamma * cfg.emf / (cfg.r_hi + gamma)
        weight = 1 / gamma
        warm = None if cfg.device == "1s1r" else np.zeros(cfg.p * cfg.q)
        a = np.zeros(cfg.num_branches)
        w = np.zeros(cfg.num_nodes)  # node potentials of M a
        v = np.zeros(cfg.num_branches)
        kcl, kvl, fixed_point, newton_steps = [], [], [], 0
        converged = False
        t = time.perf_counter()
        for k in range(self.max_iterations):
            # 1) scattering at the branches
            v_dev, steps, warm = cell_resolvent(cfg, a[d], gamma[d], warm)
            newton_steps += steps
            v = lin_gain * a + lin_offset
            v[d] = v_dev
            b = 2 * v - a
            i = (a - b) / (2 * gamma)

            # 2) scattering at the network: M b = A^T u_b
            u_b = solve_K(A @ (b * weight))
            Sb = 2 * (AT @ u_b) - b
            r = Sb - a
            fixed_point.append(float(np.sqrt((r * r) @ weight / max((a * a) @ weight, (Sb * Sb) @ weight, 1e-300))))

            # (v, i) against KCL and KVL, through the node potentials u of M v = (M a + M b) / 2
            u = (w + u_b) / 2
            # 1S1R cells: warm-start the test's series solve from this iteration's cell states
            v_u, _, kcl_ok, kcl_abs, kcl_rel = kcl_check(cfg, u, self.tol, abstol, A,
                                                         warm if cfg.device == "1s1r" else None)
            v_max, kvl_abs = np.abs(v).max(), np.abs(v - v_u).max()
            kcl.append(kcl_rel)
            kvl.append(float(kvl_abs / v_max) if v_max > 0 else 0.0)
            if kcl_ok and kvl_abs <= self.tol * v_max:
                converged = True
                break

            a = (1 - alpha) * a + alpha * Sb
            w = (1 - alpha) * w + alpha * u_b
        history["iteration_time"] = time.perf_counter() - t
        history.update(kcl_residual=kcl_abs, kcl_residual_rel=kcl_rel, kvl_residual_rel=kvl[-1],
                       newton_steps=newton_steps, kcl_history=kcl, kvl_history=kvl, fixed_point_history=fixed_point)
        return SolveOutput(v, i, converged, k + 1, history)


class SpingarnSectorSolver(SpingarnSolver):
    """SpingarnSolver with the devices matched over their whole sector (gamma="geometric"), which minimizes the a priori
    contraction bound whatever the operating point, at the cost of more iterations on the default crossbar."""
    name = "spingarn_sector"
    default_gamma = "geometric"


class SpingarnSLUSolver(SpingarnSolver):
    """SpingarnSolver on its original linear solver, kept for comparison: SuperLU's LU in minimum-degree ordering."""
    name = "spingarn_slu"
    default_factorization = "slu"
    default_ordering = "builtin"


class SpingarnCholmodSolver(SpingarnSolver):
    """SpingarnSolver with CHOLMOD left to choose its own ordering (AMD in cvxopt's build), which isolates what the
    geometric nested-dissection ordering adds."""
    name = "spingarn_cholmod"
    default_ordering = "builtin"
