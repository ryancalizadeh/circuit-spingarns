"""
Laws of the crossbar's cells, as elementwise functions of per-cell parameters, shared by Config (cell currents and
conductances) and the Spingarn solver (the cells' resolvents).

    memristor  i = g_m(v) = v / R_max + (1 / R_min - 1 / R_max) tanh(v), incremental resistance in [R_min, R_max];
    selector   i = g_s(v) = v / R_off + (1 / R_on - 1 / R_off) (r(v - V_th) - r(-v - V_th)), a symmetric threshold
               selector. Its incremental conductance is 1 / R_off for |v| <= V_th - delta, 1 / R_on for
               |v| >= V_th + delta, and ramps linearly in between: r is the ramp max(x, 0) rounded quadratically over
               |x| < delta, so delta sets only the sharpness of the knee, and delta = 0 is the piecewise-linear selector
               with a kink at |v| = V_th.

    Shockley   i = g_d(v) = I_s (e^{v / c} - 1) with c = n V_T, a pn or Schottky diode: incremental resistance
               c / (i + I_s), from ~0 in deep forward bias to infinity in deep reverse bias, so its sector is [0, inf).

A 1S1R cell is a selector in series with a memristor: at cell voltage v = x_s + x_m it carries i = g_s(x_s) = g_m(x_m),
and its incremental resistance, the sum of the two, lies in [R_on + R_min, R_off + R_max]. Its law has no closed form;
series_solve computes it. A lumped Shockley 1D1R cell is a Shockley diode in series with a memristor, i = g_d(x_d) =
g_m(x_m) at v = x_d + x_m, with incremental resistance in [R_min, inf); diode_series_solve computes it. A Shockley diode
alone has a closed-form resolvent, shockley_resolvent.
"""
import numpy as np
from scipy.special import wrightomega


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


def shockley_current(v, I_s, c):
    """The Shockley diode, i = I_s (e^{v / c} - 1) with c = n V_T; expm1 keeps it accurate near v = 0."""
    return I_s * np.expm1(v / c)


def shockley_conductance(v, I_s, c):
    """Incremental conductance I_s e^{v / c} / c: 0 in deep reverse bias (underflow), unbounded in forward bias."""
    return I_s / c * np.exp(v / c)


def shockley_resolvent(a, gamma, I_s, c):
    """
    A Shockley diode behind a port resistance gamma > 0, elementwise: the root x of x + gamma I_s (e^{x / c} - 1) = a.

    With s = gamma I_s, the substitution x = a + s - c w turns the equation into w + ln w = L, L = (a + s) / c + ln(s / c),
    whose root is the Wright omega function w = omega(L). omega(L) ~ L - ln L for large L and ~ e^L for very negative L,
    and scipy evaluates it without overflow for any L, so the formula holds for any a and gamma, e.g. a = 1e5 V at gamma =
    1e8 ohm, and |x| <= |a|. But x is the difference of a + s and c omega, which cancels whenever |x| is small against
    them: relative residuals up to ~1e-3 measured, and 100% for a ~ 0. So for |a| < c the start is instead the
    linearization a / (1 + s / c), which lies at or right of the root (expm1(y) >= y), from where Newton's method on the
    convex left side descends monotonically. Four Newton steps from either start reach rounding: relative residual
    <= 1.1e-14 measured for |a| in [1e-300, 1e5] V, gamma in [1e-6, 1e10] ohm and I_s in [1e-14, 1e-3] A. The steps
    cannot overflow: e^{x / c} = 1 + (a - x) / s is finite near the root, and the omega start lies within rounding of it.
    """
    s = gamma * I_s
    with np.errstate(over="ignore", invalid="ignore"):  # the branch np.where discards may overflow
        x = np.where(np.abs(a) < c, a / (1 + s / c), a + s - c * wrightomega((a + s) / c + np.log(s / c)))
    for _ in range(4):
        x = x - (x + s * np.expm1(x / c) - a) / (1 + s / c * np.exp(x / c))
    return x


def diode_series_solve(a, gamma, R_min, R_max, I_s, c, x0=None):
    """
    A lumped Shockley 1D1R cell (a Shockley diode in series with a memristor) behind a port resistance gamma >= 0,
    elementwise: the diode voltage x_d, memristor voltage x_m and current i with i = I_s (e^{x_d / c} - 1) = g_m(x_m) and
    x_d + x_m + gamma i = a. gamma = 0 gives the cell's law at cell voltage a, gamma > 0 its resolvent J_{gamma T}(a).
    x0 = (x_d, x_m) is an optional warm start. Returns the cell voltage v = x_d + x_m, i, x_d, x_m and the outer steps.

    The diode is not odd, so the unknown is the signed diode voltage t = x_d. H(t) = t + g_m^-1(i(t)) + gamma i(t) is
    increasing with slope 1 + g_d'(t) (1 / g_m'(x_m) + gamma), and H(t) = a has its root in
        a >= 0:  [0, min(a, c log1p(a / ((R_min + gamma) I_s)))], since g_m^-1(i) >= R_min i for i >= 0, so the current
                 is at most a / (R_min + gamma);
        a < 0:   [a, min(0, a + (R_max + gamma) I_s)], since i > -I_s and |g_m^-1(i)| <= R_max |i|.
    The forward bound keeps e^{t / c} finite however large a is (gamma up to 1e8 ohm puts a at ~1e5 V under adaptive
    matching). The solve is series_solve's: safeguarded Newton (rtsafe) from the bracket's upper end (H is convex in
    forward bias, so Newton descends monotonically from there) or the warm start, g_m^-1 by tanh_root, and cells retire
    as they converge.
    """
    a = np.asarray(a, dtype=float)
    gamma, R_min, R_max = (np.broadcast_to(np.asarray(x, dtype=float), a.shape) for x in (gamma, R_min, R_max))
    fwd = a >= 0
    with np.errstate(divide="ignore"):  # gamma = 0 and R_min > 0: never 0 / 0
        lo = np.where(fwd, 0.0, a)
        hi = np.where(fwd, np.minimum(a, c * np.log1p(np.maximum(a, 0.0) / ((R_min + gamma) * I_s))),
                      np.minimum(0.0, a + (R_max + gamma) * I_s))
    if x0 is None:
        t, x_m = hi.copy(), np.zeros_like(a)
    else:
        t, x_m = np.clip(x0[0], lo, hi), np.array(x0[1], dtype=float)
    i = np.zeros_like(a)

    def evaluate(k, t, x_m):
        """H(t) - a, its slope, the current, the memristor voltage and convergence, for the cells k."""
        g, Rn, Rx = gamma[k], R_min[k], R_max[k]
        e = np.exp(t / c)
        i = I_s * np.expm1(t / c)
        x_m, _ = tanh_root(1 / Rx, 1 / Rn - 1 / Rx, i, x_m)
        f = t + x_m + g * i - a[k]
        df = 1 + I_s / c * e * (1 / memristor_conductance(x_m, Rn, Rx) + g)
        return f, df, i, x_m, np.abs(f) <= 1e-13 * (np.abs(t) + np.abs(x_m) + g * np.abs(i) + np.abs(a[k]))

    k = np.arange(a.size)
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
            return t + x_m, i, t, x_m, steps - 1
        lo_k = np.where(f < 0, t_k, lo_k)
        hi_k = np.where(f > 0, t_k, hi_k)
        bisect = (((t_k - hi_k) * df - f) * ((t_k - lo_k) * df - f) > 0) | (np.abs(2 * f) > np.abs(dx_old * df))
        dx_old = dx
        dx = np.where(bisect, 0.5 * (hi_k - lo_k), f / df)
        t_k = np.where(bisect, lo_k + dx, t_k - dx)
        f, df, i_k, x_k, done = evaluate(k, t_k, x_k)
        done |= np.abs(dx) <= 1e-15 * np.abs(t_k)
    raise RuntimeError("diode_series_solve: did not converge")
