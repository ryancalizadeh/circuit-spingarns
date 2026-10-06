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

In the scaled waves Gamma^{-1/2} a, S is an orthogonal reflection, and the branch reflection a_e -> b_e is Lipschitz with
constant max |R - gamma_e| / (R + gamma_e) over branch e's incremental resistances R. With alpha = 1 the iteration is
therefore a contraction by the worst such factor, whatever the size or topology of the array. Impedance matching chooses
Gamma to make the factors small: gamma_e = R absorbs a linear branch's wave completely.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Config import BranchKind
from algorithms.common import Solver, SolveOutput, kcl_check

GAMMA_RULES = ("geometric", "r_lo")


def port_resistances(cfg, rule):
    """
    Gamma's diagonal, one port resistance per branch. Linear branches get their own resistance. Devices, whose
    incremental resistance ranges over [R_min, R_max], get by `rule`:

        "geometric"  sqrt(R_min R_max), which minimizes the worst reflection factor over that whole range;
        "r_lo"       R_min, their incremental resistance at zero bias;
        a number     that resistance in ohms.
    """
    gamma = cfg.r_hi.copy()  # r_lo = r_hi = R on linear branches
    d = cfg.slices[BranchKind.DEVICE]
    if rule == "geometric":
        gamma[d] = np.sqrt(cfg.R_min * cfg.R_max)
    elif rule == "r_lo":
        gamma[d] = cfg.R_min
    else:
        gamma[d] = float(rule)
    return gamma


def contraction_bound(cfg, gamma, alpha):
    """A priori bound on the factor by which each iteration shrinks the distance to the fixed point (see above)."""
    rho = np.maximum(np.abs(cfg.r_lo - gamma) / (cfg.r_lo + gamma), np.abs(cfg.r_hi - gamma) / (cfg.r_hi + gamma))
    return float(1 - alpha + alpha * rho.max())


def device_resolvent(a, gamma, R_min, R_max, v0):
    """
    J_{gamma T}(a) for devices, elementwise: the root v of c v + d tanh(v) = a with c = 1 + gamma / R_max and
    d = gamma (1 / R_min - 1 / R_max), by Newton's method from the warm start v0. Returns v and the number of steps.

    The root has the sign of a, and x = |v| lies in [lo, hi] below. The left side is concave in x >= 0, so a Newton step
    from anywhere in [lo, hi], clipped to it, lands at or below the root, and from there the steps rise monotonically to
    it.
    """
    c = 1 + gamma / R_max
    d = gamma * (1 / R_min - 1 / R_max)
    sign, target = np.sign(a), np.abs(a)
    lo = np.maximum(target / (c + d), (target - d) / c)  # tanh x <= x and tanh x < 1
    hi = target / c  # tanh x >= 0
    x = np.clip(np.abs(v0), lo, hi)
    for steps in range(1, 101):
        t = np.tanh(x)
        x_new = np.clip(x - (c * x + d * t - target) / (c + d * (1 - t * t)), lo, hi)
        done = np.all(np.abs(x_new - x) <= 1e-12 * x_new)  # convergence is quadratic: x_new is exact to rounding
        x = x_new
        if done:
            return sign * x, steps
    raise RuntimeError("device resolvent: Newton's method did not converge")


class SpingarnSolver(Solver):
    """
    Spingarn's method of partial inverses with impedance-matching preconditioning (see the module docstring).

    gamma picks the devices' port resistance (see port_resistances); None selects the class's default_gamma. This class
    matches the devices at zero bias, where the crossbar's devices mostly operate; SpingarnSectorSolver matches them over
    their whole sector. alpha in (0, 1] is the relaxation: 1/2 is Spingarn's original method, and 1 contracts fastest
    when every branch is strictly increasing, as here. abstol is the absolute current tolerance of the KCL test, as in
    SPICE.

    The iterate (v, i) satisfies the branch laws exactly, so convergence is judged on the constraints it meets only in
    the limit, both to relative tolerance tol: KCL by SPICE's own test (kcl_check) on the node potentials u of v, which
    the projection yields at no extra solve, and KVL by max |v - A^T u| <= tol * max |v|. A converged answer thus passes
    the same KCL test as SPICE's. Iterations count passes of the loop, i.e. linear solves.
    """
    name = "spingarn"
    default_gamma = "r_lo"

    def __init__(self, tol=None, max_iterations=None, gamma=None, alpha=1.0, abstol=1e-12):
        gamma = self.default_gamma if gamma is None else gamma
        if not (gamma in GAMMA_RULES if isinstance(gamma, str) else isinstance(gamma, (int, float)) and gamma > 0):
            raise ValueError(f"gamma must be one of {GAMMA_RULES} or a positive resistance, not {gamma!r}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must lie in (0, 1], not {alpha!r}")
        super().__init__(tol, max_iterations, gamma=gamma, alpha=alpha, abstol=abstol)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def solve(self, cfg):
        alpha, abstol = self.options["alpha"], self.options["abstol"]
        history = {}
        t = time.perf_counter()
        A = cfg.incidence()
        AT = A.T.tocsr()
        gamma = port_resistances(cfg, self.options["gamma"])
        lu = spla.splu((A @ sp.diags(1 / gamma) @ AT).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0,
                       options=dict(SymmetricMode=True))
        history["setup_time"] = time.perf_counter() - t
        history["factor_nnz"] = int(lu.nnz)
        history["contraction_bound"] = contraction_bound(cfg, gamma, alpha)

        d = cfg.slices[BranchKind.DEVICE]
        lin_gain = cfg.r_hi / (cfg.r_hi + gamma)  # J_{gamma T}(a) = lin_gain * a + lin_offset on linear branches
        lin_offset = gamma * cfg.emf / (cfg.r_hi + gamma)
        weight = 1 / gamma
        a = np.zeros(cfg.num_branches)
        w = np.zeros(cfg.num_nodes)  # node potentials of M a
        v = np.zeros(cfg.num_branches)
        kcl, kvl, fixed_point, newton_steps = [], [], [], 0
        converged = False
        t = time.perf_counter()
        for k in range(self.max_iterations):
            # 1) scattering at the branches
            v_dev, steps = device_resolvent(a[d], gamma[d], cfg.R_min, cfg.R_max, v[d])
            newton_steps += steps
            v = lin_gain * a + lin_offset
            v[d] = v_dev
            b = 2 * v - a
            i = (a - b) / (2 * gamma)

            # 2) scattering at the network: M b = A^T u_b
            u_b = lu.solve(A @ (b * weight))
            Sb = 2 * (AT @ u_b) - b
            r = Sb - a
            fixed_point.append(float(np.sqrt((r * r) @ weight / max((a * a) @ weight, (Sb * Sb) @ weight, 1e-300))))

            # (v, i) against KCL and KVL, through the node potentials u of M v = (M a + M b) / 2
            u = (w + u_b) / 2
            v_u, _, kcl_ok, kcl_abs, kcl_rel = kcl_check(cfg, u, self.tol, abstol, A)
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
