"""
Laws of the crossbar's cells, as elementwise functions of per-cell parameters, shared by Config (cell currents and
conductances) and the Spingarn solver (the cells' resolvents).

    memristor  i = g_m(v) = v / R_max + (1 / R_min - 1 / R_max) tanh(v), incremental resistance in [R_min, R_max];
    selector   i = g_s(v) = v / R_off + (1 / R_on - 1 / R_off) (r(v - V_th) - r(-v - V_th)), a symmetric threshold
               selector. Its incremental conductance is 1 / R_off for |v| <= V_th - delta, 1 / R_on for
               |v| >= V_th + delta, and ramps linearly in between: r is the ramp max(x, 0) rounded quadratically over
               |x| < delta, so delta sets only the sharpness of the knee, and delta = 0 is the piecewise-linear selector
               with a kink at |v| = V_th.

A 1S1R cell is a selector in series with a memristor: at cell voltage v = x_s + x_m it carries i = g_s(x_s) = g_m(x_m),
and its incremental resistance, the sum of the two, lies in [R_on + R_min, R_off + R_max]. Its law has no closed form;
series_solve computes it.
"""
import numpy as np


def memristor_current(v, R_min, R_max):
    return v / R_max + (1 / R_min - 1 / R_max) * np.tanh(v)


def memristor_conductance(v, R_min, R_max):
    return 1 / R_max + (1 / R_min - 1 / R_max) * (1 - np.tanh(v) ** 2)


def tanh_root(c, d, target, x0):
    """
    The root x of c x + d tanh(x) = target, elementwise, for c > 0 and d >= 0, by Newton's method from the warm start
    x0. Returns x and the number of steps.

    The root has the sign of the target, and |x| lies in [lo, hi] below. The left side is concave in x >= 0, so a Newton
    step from anywhere in [lo, hi], clipped to it, lands at or below the root, and from there the steps rise
    monotonically to it.
    """
    sign, target = np.sign(target), np.abs(target)
    lo = np.maximum(target / (c + d), (target - d) / c)  # tanh x <= x and tanh x < 1
    hi = target / c  # tanh x >= 0
    x = np.clip(np.abs(x0), lo, hi)
    for steps in range(1, 101):
        t = np.tanh(x)
        x_new = np.clip(x - (c * x + d * t - target) / (c + d * (1 - t * t)), lo, hi)
        done = np.all(np.abs(x_new - x) <= 1e-12 * x_new)  # convergence is quadratic: x_new is exact to rounding
        x = x_new
        if done:
            return sign * x, steps
    raise RuntimeError("tanh_root: Newton's method did not converge")


def _ramp(x, delta):
    """max(x, 0), rounded to (x + delta)^2 / (4 delta) over |x| < delta."""
    if delta == 0:
        return np.maximum(x, 0.0)
    return np.where(x >= delta, x, np.where(x > -delta, (x + delta) ** 2 / (4 * delta), 0.0))


def _ramp_slope(x, delta):
    if delta == 0:
        return (x > 0).astype(float)
    return np.clip((x + delta) / (2 * delta), 0.0, 1.0)


def selector_current(v, R_on, R_off, V_th, delta):
    return v / R_off + (1 / R_on - 1 / R_off) * (_ramp(v - V_th, delta) - _ramp(-v - V_th, delta))


def selector_conductance(v, R_on, R_off, V_th, delta):
    """Incremental conductance, within [1 / R_off, 1 / R_on]; at the kink of delta = 0, the off value."""
    return 1 / R_off + (1 / R_on - 1 / R_off) * (_ramp_slope(v - V_th, delta) + _ramp_slope(-v - V_th, delta))


def series_solve(a, gamma, R_min, R_max, R_on, R_off, V_th, delta, x0=None):
    """
    A 1S1R cell behind a port resistance gamma >= 0, elementwise: the selector voltage x_s, memristor voltage x_m and
    current i with i = g_s(x_s) = g_m(x_m) and x_s + x_m + gamma i = a. With gamma = 0 this is the cell's law at cell
    voltage a; with gamma > 0 it is the cell's resolvent J_{gamma T}(a). x0 = (x_s, x_m) is an optional warm start.
    Returns the cell voltage v = x_s + x_m, i, x_s, x_m and the number of outer steps.

    In the one unknown t = |x_s|, the equation is H(t) = |a| with H(t) = t + g_m^-1(g_s(t)) + gamma g_s(t), increasing
    with slope 1 + g_s'(t) (1 / g_m'(x_m) + gamma); the solution has the sign of a. Since g_s has slopes in
    [1 / R_off, 1 / R_on] and g_m in [1 / R_max, 1 / R_min], the root lies in
    [|a| / (1 + (R_max + gamma) / R_on), |a| / (1 + (R_min + gamma) / R_off)]. A safeguarded Newton iteration (rtsafe:
    the Newton step when it stays in the bracket and at least halves the step before last, bisection otherwise) converges
    from anywhere in it, however sharp the selector's knee; g_m^-1 is tanh_root, warm-started from the previous step.

    A cell is done when its equation holds to rounding in its own terms, or its bracket has collapsed. Done cells leave
    the iteration (a step from f ~ 0 could bisect them away from the root, and they would cost as much as the rest), so
    the work follows the cells still converging, not the slowest one. The steps returned are the rounds until the last
    cell is done.
    """
    sign, A = np.sign(a), np.abs(a)
    gamma, R_min, R_max = (np.broadcast_to(np.asarray(x, dtype=float), A.shape) for x in (gamma, R_min, R_max))
    lo = A / (1 + (R_max + gamma) / R_on)
    hi = A / (1 + (R_min + gamma) / R_off)
    if x0 is None:  # start at the threshold: the first evaluation tells on which side of the knee the root lies
        t, x_m = np.clip(V_th, lo, hi), np.zeros_like(A)
    else:
        t, x_m = np.clip(np.abs(x0[0]), lo, hi), np.abs(x0[1])
    i = np.zeros_like(A)

    def evaluate(k, t, x_m):
        """H(t) - |a|, its slope, the current, the memristor voltage and convergence, for the cells k."""
        g, Rn, Rx = gamma[k], R_min[k], R_max[k]
        i = selector_current(t, R_on, R_off, V_th, delta)
        x_m, _ = tanh_root(1 / Rx, 1 / Rn - 1 / Rx, i, x_m)
        f = t + x_m + g * i - A[k]
        df = 1 + selector_conductance(t, R_on, R_off, V_th, delta) * (1 / memristor_conductance(x_m, Rn, Rx) + g)
        return f, df, i, x_m, np.abs(f) <= 1e-13 * (t + x_m + g * i + A[k])

    # the cells still converging, k, and their state
    k = np.arange(A.size)
    t_k, x_k, lo_k, hi_k = t, x_m, lo, hi
    f, df, i_k, x_k, done = evaluate(k, t_k, x_k)
    dx_old = dx = hi_k - lo_k
    for steps in range(1, 301):
        if done.any():
            t[k[done]], x_m[k[done]], i[k[done]] = t_k[done], x_k[done], i_k[done]
            keep = ~done
            k, t_k, x_k, i_k, f, df, lo_k, hi_k, dx, dx_old = (
                arr[keep] for arr in (k, t_k, x_k, i_k, f, df, lo_k, hi_k, dx, dx_old))
        if k.size == 0:
            return sign * (t + x_m), sign * i, sign * t, sign * x_m, steps - 1
        lo_k = np.where(f < 0, t_k, lo_k)
        hi_k = np.where(f > 0, t_k, hi_k)
        bisect = (((t_k - hi_k) * df - f) * ((t_k - lo_k) * df - f) > 0) | (np.abs(2 * f) > np.abs(dx_old * df))
        dx_old = dx
        dx = np.where(bisect, 0.5 * (hi_k - lo_k), f / df)
        t_k = np.where(bisect, lo_k + dx, t_k - dx)
        f, df, i_k, x_k, done = evaluate(k, t_k, x_k)
        done |= np.abs(dx) <= 1e-15 * t_k
    raise RuntimeError("series_solve: did not converge")
