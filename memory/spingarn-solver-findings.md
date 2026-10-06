---
name: spingarn-solver-findings
description: "measured Spingarn solver behaviour on the crossbar and the user's design decisions: size-independent iteration counts, device Gamma choice, accuracy vs SPICE, SuperLU scaling"
metadata:
  type: project
---

Measured 2026-10-05 on the nominal crossbar (E 0.1-0.5 V), tol 1e-9, alpha 1:

- Iteration count does not depend on array size, 1x1 to 1024x1024: 5 with devices matched at R_min (`spingarn`), 85-92 with sqrt(R_min R_max) (`spingarn_sector`). With E up to 1.5 V / 3 V: 7-8 / 11-15 vs 81-87. A priori bounds 0.98 vs 0.818; observed rates ~0.006 vs ~0.796.
- Matching devices to their provable operating range, sqrt(R_inc(0) R_inc(max E)) via the no-gain property, was tried and is worse than R_min in practice (7-61 iterations): the count follows the aggregate reflection, and most devices sit near 0 V in the IR-drop region.
- alpha = 1 takes about half the iterations of Spingarn's original 1/2; strongly monotone branches make the iteration a contraction.
- Runtime at scale is the single SuperLU factorization of A Gamma^-1 A^T: 1024x1024 factors in 29 s, 270M stored nonzeros, 3.75 GB peak; totals 32 s (`spingarn`) and 78 s (`spingarn_sector`) vs SPICE/KLU's projected 6-7 min and 12-15 GB. At 256x256 the triangular solve is only ~30% of an iteration, the rest numpy and KCL-check overhead (not yet optimized). 2048x2048 untested: ~18 GB projected, and SuperLU's indices are 32-bit.
- SPICE at RELTOL 1e-3 actually returns KCL residuals ~1e-11 (Newton overshoots); Spingarn stops at its tol (~1e-9). At tol 1e-9 the KCL test's abstol 1e-12 A binds at nodes carrying < 1 mA, so converged runs can report kcl_residual_rel up to ~1e-6 at 1024x1024.

User decision (2026-10-05): register both device matchings as separate solvers, `spingarn` (r_lo) and `spingarn_sector` (geometric), so one sweep compares them with SPICE.

**Why:** these are the evidence for the size-independence hypothesis in [[project-goals]] and decide how Spingarn vs SPICE comparisons must be read alongside [[ngspice-baseline-findings]].

**How to apply:** compare runtimes at matched accuracy (check both solvers' kcl_residual_rel, align tolerances); say which variant a number came from; attribute large-size runtime to the factorization, not the iteration count.
