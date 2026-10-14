"""
Newton's method on the crossbar's nodal KCL equations, on the Spingarn solver's linear algebra.

The unknowns are the node potentials u, and KCL is F(u) = A g(A^T u) = 0 with Jacobian J = A G A^T, G the branches'
incremental conductances. Each step solves J du = F and sets u <- u - du. J has the sparsity pattern of the nodal matrix
A Gamma^-1 A^T that Spingarn factorizes, and is symmetric positive definite whenever every conductance is >= 0 and every
node reaches ground through positive ones (the wires, sources and loads see to that), so it is factorized the same way:
CHOLMOD's supernodal Cholesky in the geometric nested-dissection ordering (algorithms/ordering.py), analysed once, then
only the numeric factorization redone each step (solve.refactor). This is the baseline the comparison with SPICE lacked
(report_adaptive_matching.md section 4): the same Newton iteration SPICE runs, on Spingarn's factorization and without
ngspice's overheads, and without its device limiting (pnjlim), its gmin or its fallbacks.

Plain Newton is undamped, like SPICE's. With safeguard=True each step is halved until the 2-norm of the KCL residual
decreases (at most max_halvings times, after which the shortest step is taken anyway): a cheap guard against the 2-cycles
plain Newton can fall into at a kink and against the overshoot of exponential laws. Each trial point costs one evaluation
of the branch laws; a full step that decreases the residual costs nothing extra.

Convergence is SPICE's KCL test (kcl_check) on the iterate, to relative tolerance tol and absolute tolerance abstol, the
test every solver here is judged by. Iterations count Newton steps, i.e. factorizations. A residual that overflows, or a
Jacobian that CHOLMOD cannot factorize, ends the run unconverged with the reason in history["failure"].
"""
import time

import numpy as np
import scipy.sparse as sp

from Config import BranchKind
from devices import shockley_conductance, shockley_current
from algorithms.common import Solver, SolveOutput, kcl_check
from algorithms.ordering import nested_dissection
from algorithms.spingarns import ORDERINGS, blas_threads, cell_conductance, factorize


class NodalMatrix:
    """
    A G A^T for branch conductances G on a sparsity pattern fixed once: every entry any branch can touch is stored, with
    explicit zeros where the conductances happen to vanish (a reverse-biased 1D1R cell, an underflowed diode), so that
    each step's matrix has the pattern CHOLMOD analysed. Forming A diag(g) A^T by sparse products would drop those zeros
    and change the pattern.
    """

    def __init__(self, cfg, A):
        pattern = (abs(A) @ abs(A).T).tocsr()
        pattern.sort_indices()
        self.indptr, self.indices, self.shape = pattern.indptr, pattern.indices, pattern.shape
        # each branch adds g to its two diagonal entries and -g to its two off-diagonal ones (ground has no row); an
        # entry's slot in the CSR data is found by its row-major key, which the sorted pattern stores in increasing order
        n = self.shape[0]
        keys = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.indptr)) * n + self.indices
        slot = lambda r, c: np.searchsorted(keys, r.astype(np.int64) * n + c)
        k = np.arange(cfg.num_branches)
        t, h = cfg.tail, cfg.head
        both = (t >= 0) & (h >= 0)
        self.positions = np.concatenate([slot(t[t >= 0], t[t >= 0]), slot(h[h >= 0], h[h >= 0]),
                                         slot(t[both], h[both]), slot(h[both], t[both])])
        self.branches = np.concatenate([k[t >= 0], k[h >= 0], k[both], k[both]])
        self.signs = np.concatenate([np.ones((t >= 0).sum() + (h >= 0).sum()), -np.ones(2 * both.sum())])

    def __call__(self, g):
        data = np.bincount(self.positions, weights=self.signs * g[self.branches], minlength=self.indices.size)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=self.shape)


def branch_laws(cfg, v, warm):
    """
    Currents and incremental conductances of all of cfg's branches at branch voltages v, and the warm start for the next
    call: series cells (1S1R, lumped Shockley 1D1R) are solved once for both, warm-started from their previous inner
    voltages. The currents are Config.current's.
    """
    i = (v - cfg.emf) / cfg.r_hi
    g = 1 / cfg.r_hi
    d = cfg.slices[BranchKind.DEVICE]
    if cfg.device_law == "1s1r":
        _, i[d], v_s, v_m, _ = cfg.series_solve(v[d], 0.0, warm)
        warm = (v_s, v_m)
        g[d] = cell_conductance(cfg, v[d], warm)
    elif cfg.device_law == "1d1r_shockley":
        _, i[d], v_d, v_m, _ = cfg.diode_series_solve(v[d], 0.0, warm)
        warm = (v_d, v_m)
        g[d] = cfg.diode_series_conductance(v_d, v_m)
    else:
        i[d], g[d] = cfg.device_current(v[d]), cfg.device_conductance(v[d])
    if cfg.diode_law == "shockley":
        e = cfg.slices[BranchKind.DIODE]
        c = cfg.n * cfg.V_T
        i[e], g[e] = shockley_current(v[e], cfg.I_s, c), shockley_conductance(v[e], cfg.I_s, c)
    return i, g, warm


class NewtonSolver(Solver):
    """
    Newton's method on nodal KCL (see the module docstring), undamped. safeguard=True halves steps that fail to
    decrease the KCL residual, at most max_halvings times per step. ordering ("nested_dissection" or "builtin", CHOLMOD's
    own AMD) and blas_threads as for SpingarnSolver. Cannot solve circuits with ideal diodes on branches of their own,
    which have no finite conductance.
    """
    name = "newton"
    default_max_iterations = 200
    default_safeguard = False

    def __init__(self, tol=None, max_iterations=None, safeguard=None, max_halvings=30, abstol=1e-12,
                 ordering="nested_dissection", blas_threads=1):
        safeguard = self.default_safeguard if safeguard is None else safeguard
        if not isinstance(safeguard, bool):
            raise ValueError(f"safeguard must be True or False, not {safeguard!r}")
        if not (type(max_halvings) is int and max_halvings >= 1):
            raise ValueError(f"max_halvings must be a positive integer, not {max_halvings!r}")
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, not {ordering!r}")
        if not (blas_threads is None or type(blas_threads) is int and blas_threads >= 1):
            raise ValueError(f"blas_threads must be a positive integer or None, not {blas_threads!r}")
        super().__init__(tol, max_iterations, safeguard=safeguard, max_halvings=max_halvings, abstol=abstol,
                         ordering=ordering, blas_threads=blas_threads)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def solve(self, cfg):
        if cfg.diode_law == "ideal":
            raise ValueError("newton needs finite conductances; an ideal diode on its own branch has none")
        with blas_threads(self.options["blas_threads"]) as threads:
            return self._solve(cfg, threads)

    def _solve(self, cfg, threads):
        o = self.options
        history = dict(blas_threads=threads)
        t = time.perf_counter()
        A = cfg.incidence()
        perm = nested_dissection(cfg) if o["ordering"] == "nested_dissection" else None
        jacobian = NodalMatrix(cfg, A)
        history["setup_time"] = time.perf_counter() - t

        state = {}

        def laws(v, warm):  # kcl_check's branch laws, keeping the conductances and warm start they come with
            i, state["g"], state["warm"] = branch_laws(cfg, v, warm)
            return i

        def evaluate(u, warm):
            """KCL test at u: (v, i, ok, |residual|_max, rel), the residual's 2-norm, and the laws' by-products."""
            with np.errstate(over="ignore", invalid="ignore"):  # overflow is caught below, as a non-finite residual
                check = kcl_check(cfg, u, self.tol, o["abstol"], A, warm, laws)
                F = A @ check[1]
                norm = float(np.sqrt(F @ F)) if np.all(np.isfinite(F)) else np.inf
            return check, F, norm, state["g"], state["warm"]

        solve_K, factor_time, eval_time = None, 0.0, 0.0
        kcl, step_max, halvings_per_step = [], [], []
        failure, converged, k = None, False, 0
        t = time.perf_counter()
        u = np.zeros(cfg.num_nodes)
        t_eval = time.perf_counter()
        (v, i, ok, kcl_abs, kcl_rel), F, norm, g, warm = evaluate(u, None)
        eval_time += time.perf_counter() - t_eval
        kcl.append(kcl_rel)
        while True:
            if ok:
                converged = True
                break
            if not np.isfinite(norm) or not np.all(np.isfinite(g)):
                failure = "non-finite residual or conductance"
                break
            if k >= self.max_iterations:
                break
            t_factor = time.perf_counter()
            J = jacobian(g)
            try:
                if solve_K is None:
                    solve_K, info = factorize(J, "cholmod", perm)
                    history.update(info)
                else:
                    solve_K.refactor(J)
            except ArithmeticError as err:  # CHOLMOD: not positive definite (e.g. an underflowed node)
                failure = f"factorization failed: {err}"
                break
            du = solve_K(F)
            factor_time += time.perf_counter() - t_factor
            k += 1
            step_max.append(float(np.abs(du).max()))
            step, halvings = 1.0, 0
            t_eval = time.perf_counter()
            while True:
                trial = evaluate(u - step * du, warm)
                if not o["safeguard"] or trial[0][2] or trial[2] < norm or halvings >= o["max_halvings"]:
                    break
                step, halvings = step / 2, halvings + 1
            eval_time += time.perf_counter() - t_eval
            halvings_per_step.append(halvings)
            u = u - step * du
            (v, i, ok, kcl_abs, kcl_rel), F, norm, g, warm = trial
            kcl.append(kcl_rel)
        history.update(iteration_time=time.perf_counter() - t, factor_time=factor_time, eval_time=eval_time,
                       kcl_residual=kcl_abs, kcl_residual_rel=kcl_rel, kcl_history=kcl, step_max_history=step_max,
                       halvings=int(sum(halvings_per_step)), halvings_history=halvings_per_step, failure=failure)
        finite = np.all(np.isfinite(v)) and np.all(np.isfinite(i))
        return SolveOutput(v if finite else None, i if finite else None, converged, k, history)


class NewtonSafeguardedSolver(NewtonSolver):
    """NewtonSolver with step halving on by default (see the module docstring)."""
    name = "newton_safeguarded"
    default_safeguard = True
