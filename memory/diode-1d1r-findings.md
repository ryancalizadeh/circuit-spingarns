---
name: diode-1d1r-findings
description: "1D1R (ideal diode + memristor, no leak) under Spingarn: alpha=1 fails (eigenvalues near -1), size independence lost (lambda_max -> 1 on all-reverse rows), fixed by matching reverse cells to an open circuit; tanh nonlinearity not exercised at >= 128x128"
metadata:
  type: project
---

Measured 2026-10-09 on the PC (see [[measurement-machine]]) with `diode_experiment.py`, circuit `device="1d1r"`, E in [-1, 1.5] V, R_min log-uniform [1k, 10k], R_max [50k, 200k] per cell, gamma = R_min per cell (`r_lo`), tol 1e-9, 3 seeds per size, results/diode-sweep-1.

- Model: cell law i = g_m(max(v, 0)), sector [R_min, inf]; a priori bound exactly 1. Resolvent: v = a for a <= 0. SPICE uses `uramp`; Newton, Spingarn and SPICE agree.
- alpha = 1 never converges in 3000 iterations at any size >= 8x8: S D has eigenvalues at -0.999. A reverse cell reflects +1, while the network seen from its port (wires, a few ohms) reflects (R_th - gamma)/(R_th + gamma) ~ -1. The local "no unimodular eigenvalue" argument holds but gives no gap. alpha < 1 maps -1 to 1 - 2 alpha.
- Size independence is lost for fixed matching: lambda_max of S D = 0.20-0.34 (8x8), 0.61-0.65 (32), 0.83 (64), 0.935 (128), 0.979 (256). Iterations at alpha 0.5 / 0.8: ~45 / 36 (8x8), ~97 / 57 (32), ~208 / 127 (64), ~536 / 332 (128), ~1550 / 968 (256): roughly x2.8 per x4 cells. The slow mode lies on rows whose cells are all reverse-biased: the solve models each open cell as gamma = R_min, and q of them in parallel swamp the 50 ohm source.
- Lanczos rho(J) = max |1 - alpha + alpha lam| over the extreme eigenvalues of D^1/2 S~ D^1/2 matches the observed rate to 4 digits for alpha < 1 (D >= 0 under r_lo, so the spectrum is real).
- Oracle matching (gamma = 1e6 on the cells that end reverse-biased) brings lambda_max to 0.015 / 0.007 / 0.005 at 32 / 64 / 128 and rho(alpha=0.5) to ~0.50 (~30 iterations), flat in size. So the size dependence is a matching problem, not intrinsic to the kink. The practical version, adaptive (SIM-style) re-matching with refactorization, is untested.
- Nonlinearity check: diodes change the output currents by 50-110% at every size. The tanh does not: forward-cell median voltage 0.35 V (8x8) -> 0.18 (32) -> 0.08 (64) -> 0.03 (128) -> 0.009 V (256). Linearizing the memristor changes outputs 3-5% at 8x8, 0.5% at 32, <0.1% at >= 64, so large arrays here test the kink, not the tanh. The cause is loading by R_source = 50 ohm and R_load = 1 kohm against 1-10 kohm cells.
- vs SPICE (2026-10-09, PC, runner, results/diode-spice-1, same circuits and seeds 0-2, spingarn alpha 0.8 / tol 1e-9, SPICE RELTOL 1e-3 KLU): SPICE converges by plain Newton at every size (no fallbacks), but its Newton count grows slowly: 6, 6, 7, 8.7, 10.7, 12.3 at 8-256 (memristor-only: flat 4). Runtime: Spingarn is faster up to 64x64 (0.19 vs 0.27 s), crosses over at ~64-128, and is slower at 128 (2.2 vs 1.4 s) and 256 (29-48 vs 9.8 s). This reverses the memristor-only result (57x faster at 768). Memory at 256: 0.22 vs 0.88 GB. Solutions agree (max |dv| 9e-8 V, outputs 3e-8 at 256).
- Accuracy is NOT matched on 1D1R. On nodes carrying > 1 nA, SPICE's KCL rel residual is 2e-11 to 1e-9 vs Spingarn's 3e-9 to 3e-6, so the runtime gap understates SPICE's lead. The recorded kcl_residual_rel (~0.6-0.97 for spingarn) is an artifact: 27-36% of nodes sit on dead lines (all cells off, < 1 nA), where residual / max(scale, abstol) ~ 1 at a 1e-12 A residual.
- Spingarn 256 per-iteration time varied 30 vs 48 ms between seeds in one sweep (same iterations, same factor time); cause unknown.

**Why:** these decide how 1D1R results can be claimed. Without adaptive matching, Spingarn on 1D1R is not size-independent, and the default circuit does not exercise the device nonlinearity at scale.

**How to apply:** use alpha < 1 for leakless 1D1R. Quote iteration counts together with the matching rule. Don't claim tanh nonlinearity at >= 64x64 for this circuit. Check `operating_point` diagnostics whenever circuit parameters change. See [[spingarn-solver-findings]] for the memristor-only baseline (size-independent, 5 iterations).
