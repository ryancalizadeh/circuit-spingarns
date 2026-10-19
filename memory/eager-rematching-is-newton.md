---
name: eager-rematching-is-newton
description: "re-matching every iteration to the exact tangent (alpha 1) is Newton with a resolvent (diagonal) projection back onto the device curves instead of Newton's vertical one: same linear system per step, locally quadratic; fixed-Gamma Spingarn is the matching chord method. Checked 2026-10-15, seed 0, 32/64"
metadata:
  type: project
---

Derivation: the network solve with port resistances Gamma and b = v - Gamma i solves the linear circuit i' = i + (v' - v)/gamma per branch. With gamma = 1/g'(v) at a branch point on the curve, that is Newton's linearized circuit, so the matrix A G A^T and the step are Newton's. The only difference is the next linearization point: Newton keeps v' and re-evaluates i = g(v') (vertical projection); Spingarn at alpha = 1 lands on the curve along the line v + gamma i = const (the resolvent). Spingarn's branch points need not be KVL-consistent. Fixed-Gamma Spingarn = chord (frozen tangent) with the same resolvent projection; adaptive (every 2, threshold 0.5) sits between chord and Newton, like lazy-Jacobian/Shamanskii Newton.

Measured 2026-10-15 (in-process, not timed, so iteration counts only; seed 0, 32x32 and 64x64), eager = rematch_every=1, threshold=1e-9:
- Memristor and 1S1R (R_off 1M, kink): eager alpha=1 converges quadratically (e.g. 4e-7 -> 5e-14). It uses as many factorizations as Newton (2 vs 2; 5-6 vs 5).
- Lumped ideal 1D1R: eager alpha=1 stalls at residual ~1e-3 (the open-cell -1 mode: a' = 2v' - v_old). Eager split (1/2 on open cells, which turns their projection into Newton's vertical one) converges quadratically in 7/9 its with 6/7 refactors. Plain Newton converged in 13 at 32 and 2-cycled at 64; halving Newton took 8/9.
- Any uniform alpha < 1 (2/3) destroys quadratic convergence: linear at exactly 1/3.
- Ideal diode on its own branch: no Newton equivalent. Clipped to [1 ohm, 1e8], the diode reflects with |r| = 1 in both states, so no matching absorbs it. Eager alpha=1 and eager split fail; only alpha=2/3 converges (linear).

**Why:** this decides how "Spingarn vs Newton" is framed. Adaptive Spingarn is a Newton-family method with a different projection and a lazy Jacobian, not an unrelated iteration. That affects novelty claims ([[prior-work]]) and makes chord/Shamanskii Newton a fair baseline.

**How to apply:** when comparing to Newton, say adaptive = lazy-Jacobian Newton plus resolvent projection. Only the exact ideal diode on its own branch is outside the Newton family. Confirm on more seeds and sizes before reporting. Related: [[adaptive-matching-findings]], [[diode-models-findings]].
