---
name: spingarn-solver-findings
description: "measured Spingarn solver behaviour and the user's design decisions: size-independent iteration counts, device Gamma choice, accuracy vs SPICE, and the factorization (CHOLMOD + geometric nested dissection default since 2026-10-08, SuperLU legacy)"
metadata:
  type: project
---

Measured 2026-10-05 on the nominal crossbar (E 0.1-0.5 V), tol 1e-9, alpha 1:

- Iteration count does not depend on array size, 1x1 to 1024x1024: 5 with devices matched at R_min (`spingarn`), 85-92 with sqrt(R_min R_max) (`spingarn_sector`). With E up to 1.5 V / 3 V: 7-8 / 11-15 vs 81-87. A priori bounds 0.98 vs 0.818; observed rates ~0.006 vs ~0.796.
- Why `spingarn` beats its 0.98 bound (measured 2026-10-06, 16x16 seed 0, `convergence_rates.py`): the a priori bound uses the full sector [R_min, R_max], but device voltages at the solution are <= 0.123 V (R_inc <= 1015 ohm), so the true max reflection is 0.0075 and the linearized rate rho(S_Gamma D) at the fixed point is 0.0070 vs observed ~0.0054. Geometric is tight for the mirror reason: rho(S_Gamma D) = 0.8175 vs bound 0.8182, per-step ratio climbs to 0.81. Iterates stayed within the solution's device-voltage range throughout (no overshoot), which the no-gain argument alone does not guarantee.
- Matching devices to their provable operating range, sqrt(R_inc(0) R_inc(max E)) via the no-gain property, was tried and is worse than R_min in practice (7-61 iterations): the count follows the aggregate reflection, and most devices sit near 0 V in the IR-drop region.
- alpha = 1 takes about half the iterations of Spingarn's original 1/2; strongly monotone branches make the iteration a contraction.
- SPICE at RELTOL 1e-3 actually returns KCL residuals ~1e-11 (Newton overshoots); Spingarn stops at its tol (~1e-9). At tol 1e-9 the KCL test's abstol 1e-12 A binds at nodes carrying < 1 mA, so converged runs can report kcl_residual_rel up to ~1e-6 at 1024x1024.

Factorization (2026-10-08, on the laptop of [[measurement-machine]], 3 seeds per size, report.md §6):

- Registry since 2026-10-08: `spingarn` and `spingarn_sector` factorize with CHOLMOD (via cvxopt, see [[python-environment]]) in a geometric nested-dissection ordering (algorithms/ordering.py, leaf 64); `spingarn_slu` is the original SuperLU/MMD solver, which is what `spingarn` meant in report.md §1-5 and in anything older; `spingarn_cholmod` is CHOLMOD in its own (AMD) ordering.
- At 1024x1024: total 33.4 s (SuperLU/MMD) -> 8.61 s (SuperLU/GNDO) -> 5.42 s (CHOLMOD/GNDO); CHOLMOD/AMD 8.22 s. The ordering gives 3.9x, Cholesky a further 1.6x. Peak resident memory 4.17 -> 2.10 GB. GNDO's nonzeros of L grow as N^1.11 (N log N), AMD's as N^1.24.
- CHOLMOD never does nested dissection by itself here (no METIS in cvxopt's build); the factor's struct reports `amd` when left alone and `given` with GNDO.
- The factorization is no longer the bottleneck: at 1024x1024 `spingarn` spends 1.9 s factorizing and ~3 s in 5 iterations of 591 ms, of which the triangular solve is ~136 ms. The per-iteration O(n) overhead (report §4.2 item 5) is now the bigger lever, especially for `spingarn_sector`.
- `factor_nnz` counts L+U for SuperLU but only L (with supernodal padding) for CHOLMOD; `lnz` is CHOLMOD's exact nonzeros of L.

User decisions: register both device matchings as separate solvers (2026-10-05). CHOLMOD + geometric nested dissection becomes the default with SuperLU kept as legacy `spingarn_slu`, cvxopt as the CHOLMOD binding, SPICE left out of the factorization comparison, sizes up to 1024x1024, and `blas_threads=1` (2026-10-08).

**Why:** these are the evidence for the size-independence hypothesis in [[project-goals]] and decide how Spingarn vs SPICE comparisons must be read alongside [[ngspice-baseline-findings]].

**How to apply:** compare runtimes at matched accuracy (check both solvers' kcl_residual_rel, align tolerances); say which variant and which factorization a number came from (old "spingarn" numbers are SuperLU); attribute large-size runtime to the per-iteration overhead and the factorization, in that order, for the CHOLMOD default.
