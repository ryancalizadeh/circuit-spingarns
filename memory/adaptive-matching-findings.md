---
name: adaptive-matching-findings
description: "adaptive (SIM-style) re-matching, solvers spingarn_adaptive / spingarn_adaptive_split (2026-10-14, laptop): 1D1R iterations flat 22->31 (fixed 36->968), 1S1R 9-16 (sector 108/220), fastest solver at every size 8-256; spectrum theory confirmed; but it refactorizes ~as often as Newton, so the win over SPICE is not yet attributable to the method"
metadata:
  type: project
---

Measured 2026-10-14 on the laptop (AC, Best performance; see [[measurement-machine]]), runner, 3 seeds, sizes 8-256 (the laptop cap). Report: report_adaptive_matching.md; results/adaptive-1d1r, adaptive-1s1r-Roff100000, -Roff1000000, adaptive-spectrum.

- Rule: start at zero-bias matching; every 2 iterations re-match cells whose |reflection| against their incremental resistance at the iterate exceeds 0.5; open 1D1R cells get open_resistance 1e8; numeric-only CHOLMOD refactor (`solve.refactor`); carry the branch point over (a' = v + gamma' i); cap 50 re-matches, then fixed-Gamma theory guarantees termination. Fixed-Gamma solvers are bit-identical to before.
- Open cells create an eigenvalue at exactly -1 (wave bounces off a near-short network), so uniform alpha must be < 1: alpha = 2/3 gives rate 1/3 (predicted 0.336-0.346 vs observed 0.333-0.348, 16-128). The cells matched open equal the reverse-biased cells at the solution in every case, i.e. it finds the oracle matching of [[diode-1d1r-findings]].
- `spingarn_adaptive_split`: alpha 1 everywhere except 1/2 on open cells. About half the iterations (11-17 on 1D1R) and the same setting suits every device type, but there is no convergence proof at finite open_resistance (argument only in the open_resistance -> inf limit). It costs one extra solve per iteration.
- 256x256 runtimes: 1D1R adaptive 2.28 s / split 1.50 s vs SPICE 10.0 s vs fixed 35.7 s; 1S1R adaptive (alpha 1) 2.46 / 2.56 s vs SPICE 12.8 / 16.8 vs sector 9.8 / 19.1 (R_off 100k / 1M). 3-7x faster than SPICE at every size 8-256 on all three circuits; accuracy matched or better (split near SPICE on 1D1R, 1e-14 A).
- Caveat that decides the next step: adaptive refactorizes 5-10x per solve vs Newton's 8-13; a CHOLMOD/ND re-match costs ~73 ms at 256 vs ngspice KLU 300-450 ms per factorization plus ~3 s load/reorder. A Newton on our own factorization might match or beat adaptive (rough estimate 1-1.5 s at 256).
- Defaults were tuned on these circuit families (seed 0, 64-256): every 2 vs every 1/4 cost about equal; gamma_open 1e6-1e10 made no difference; alpha 0.6 / 0.75 cost 10-30% more.

**Why:** this was the parked next step 2 of report_nonsmooth.md; it removes Spingarn's only measured loss to SPICE (1D1R at scale) and changes the 1S1R picture of [[selector-1s1r-findings]].

**How to apply:** quote adaptive numbers with the re-match count and alpha; don't claim the speedup over SPICE is algorithmic until a Newton/chord baseline on the same CHOLMOD + nested-dissection factorization exists (report_nonsmooth next step 1, now the deciding experiment). Use alpha = 1 for spingarn_adaptive on memristor/1S1R. Position the re-matching itself as SIM's adaptation ([[prior-work]]). Related: [[spingarn-solver-findings]].
