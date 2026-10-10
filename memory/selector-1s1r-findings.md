---
name: selector-1s1r-findings
description: "1S1R (threshold selector + memristor) feasibility check: SPICE's Newton is unaffected by the selector's knee sharpness, even at the exact kink (3-9 its, no fallbacks); Spingarn sector-matched ~100-215 its, flat in delta. The model does not disadvantage Newton"
metadata:
  type: project
---

Measured 2026-10-09 on the PC (see [[measurement-machine]]) with `selector_feasibility.py`, results/selector-feasibility-1.

Model (`device="1s1r"`, devices.py): a symmetric threshold selector in series with the tanh memristor, treated as one branch. The selector's conductance is 1/R_off below V_th - delta and 1/R_on above V_th + delta, ramping linearly in between (a quadratically rounded ramp, so delta changes only the knee; delta = 0 is the exact kink). The cell sector is [R_on + R_min, R_off + R_max], which is finite. The cell law is implicit: `series_solve` uses safeguarded Newton (rtsafe) in the selector voltage, with nested `tanh_root` for g_m^-1, retiring cells from the iteration as they converge. A logistic/softplus smoothing was rejected because its exponential tail cuts the zero-bias resistance (1 Mohm -> ~70 kohm at delta = 100 mV).

Setup: R_on 1 kohm, V_th 0.5 V, R_off in {100 kohm, 1 Mohm}, delta in {100, 30, 10, 3, 1, 0} mV, memristors R_min 1-10 kohm and R_max 50-200 kohm per cell, E in [-1, 1.5] V; 1x1 and 16x16, 3 seeds; SPICE (RELTOL 1e-3, KLU), `spingarn_sector` (alpha 1), `spingarn` (r_lo = on state, alpha 1).

- SPICE converged by plain Newton in every run: 3 iterations at 1x1, 5-9 at 16x16, with no fallbacks and no trend in delta down to the kink. My hypothesis that a sharp knee would hurt Newton was wrong. Newton on a monotone network with bounded slopes (here 1/R_off .. 1/R_on) stays well-posed (SPD Jacobian) and acts like an active-set method on the pieces, so the kink costs ~nothing.
- Spingarn sector-matched: ~100 iterations (R_off 100 kohm, bound 0.837) and ~215 (R_off 1 Mohm, bound 0.920) at 16x16, flat in delta as the bound predicts. At 1x1 the counts are lower (27-130). On-state matching is much worse at 16x16: 280-320 and 1660-1890.
- The knee was exercised: at 16x16, 34-77% of selectors conduct and 1-12% sit within 10 mV of V_th. Spingarn and SPICE agree to <= 1e-9 V.

Scaling sweep (2026-10-09, PC, runner, isolated runs, 3 seeds; delta = 0; results/selector-scale-Roff100000 and -Roff1000000, plot results/selector-scale.png):

| size | R_off 100k: Spingarn / SPICE | R_off 1M: Spingarn / SPICE |
|---|---|---|
| 256 | 7.0 s (108 its) / 11.3 s (8 Newton) | 13.8 s (220) / 13.0 s (10-11) |
| 512 | 38.9 s (110) / 100.6 s (8) | 78.5 s (223) / 146.0 s (12-14) |
| 768 | 84.7 s (108) / fails | 173.5 s (223) / fails |

- Spingarn wins from 512 (2.6x and 1.9x by totals). The crossover is below 256 at R_off 100k and at ~256 at R_off 1M. Spingarn's iteration count is flat in size (it sits at the sector bound); SPICE's Newton count grows with size and with R_off (8 -> 14). Memory at 512: Spingarn 0.63 GB vs SPICE 5.2 GB.
- Accuracy is matched: worst KCL rel residual over all nodes (all carry current via R_off) is 4e-10 to 7e-9 for Spingarn and 9e-11 to 1e-8 for SPICE; solutions agree to ~1e-9 V.
- One-time costs removed from both (SPICE netlist load + KLU reorder; Spingarn setup) at 512: Spingarn 1.7x (100k) / 1.4x (1M) faster. At 256 SPICE is faster on that metric (1.1x / 1.7x).
- SPICE fails at 768 for 1S1R with the same signature as memristor-only 1024: singular matrix at the first Newton step, factor time 0, all fallbacks fail, ~333 s, ~10 GB. This is the KLU size limit arriving earlier (1.77M equations, ~2x the LU fill). 3/3 seeds at 100k and 1/1 at 1M; the user chose to skip the remaining 1M seeds.
- Implementation (2026-10-09): Spingarn's KCL check warm-starts the 1S1R series solve from the iterate's cell states (Config.current / kcl_check take x0), and devices.series_solve retires converged cells. Without the latter, the whole array iterated until the slowest cell converged: 5.4 vs 1.7 steps per iteration and 911 vs 348 ms per iteration at 512 (the unoptimized runs are kept in results/selector-scale-Roff100000/unoptimized/).

**Why:** this was proposed as the model where Newton is disadvantaged. It isn't; a bounded-slope monotone kink is benign for Newton.

**How to apply:** don't claim that nonsmooth monotone devices with bounded slopes favour Spingarn on iteration count; its 1S1R win at >= 512 comes from cheap iterations against SPICE's growing refactorization cost, and the 768 result is a SPICE implementation limit. Newton disadvantages come from exponential growth, near-singular Jacobians or non-monotone laws (see [[todo-sinh-devices]]). Related: [[diode-1d1r-findings]], [[spingarn-solver-findings]].
