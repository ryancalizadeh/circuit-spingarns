---
name: network-models-findings
description: "network models sweep 2026-10-19 (PC, 16-1024, 3 seeds): SPICE vs fixed-Gamma Spingarn (zero-bias / sector, CHOLMOD AMD) on memristor, lumped 1D1R and separate-diode 1D1R; zero-bias 12x SPICE on memristor; sector beats SPICE on lumped 1D1R to 512 but grows x3/doubling from 256; separate: both matchings identical, 2.2x slower than SPICE; SPICE never fell back"
metadata:
  type: project
---

Measured 2026-10-19 on the PC (see [[measurement-machine]]), `network_models_experiment.py`, results/network-models-1 (runs.jsonl, summary.md, runtime.csv/runs.csv + runtime_plot.tex pgfplots figure). User's spec: only SPICE, `spingarn_zero_bias` (gamma r_lo = R_min) and `spingarn_sector` (gamma sqrt(R_min R_max)); fixed Gamma, no adaptive re-matching; CHOLMOD's own ordering (AMD), not ND. My choices: one circuit for all three models (the 1D1R circuit: E -1..1.5 V, R_min 1-10k, R_max 50-200k), alpha 1 memristor / 0.8 on 1D1R, separate diodes at the R_min port in both variants, blas_threads 1, Spingarn cap 20k its. SPICE capped at 512 (KLU limit, [[ngspice-baseline-findings]]); at 1024 the user said to skip anything that would take over an hour, implemented as a power-law projection from 256 and 512 (`--runtime-limit-at 1024=3600`).

Medians (3 seeds, all converged):
- Memristor: zero-bias 20-21 its flat to 1024 (16.3 s at 1024), sector 117-119 flat (69.7 s); SPICE 5 Newton, 44.4 s at 512 -> zero-bias 12x, sector 2.6x faster at 512.
- Lumped 1D1R: zero-bias 976 / 3142 its at 256/512 (31 / 430 s; 1024 skipped, projected 99 min). Sector 115-130 its flat to 128, then 168 / 517 / 1596 (x3.1 per doubling; 5.5 / 72 / 868 s). SPICE 12 / 17 Newton, 8.9 / 118 s. Sector beats SPICE 1.6x at 512; zero-bias loses 3.6x.
- Separate 1D1R: zero-bias and sector take the *same* iterations per seed at scale (986 / 3151 at 256/512; e.g. 3029 both on seed 0): the diodes' fixed R_min port sets the slow mode, the memristor matching doesn't matter. ~577 s at 512 vs SPICE 262 s (2.2x slower); 1024 skipped (projected 130 min). Sector is slower than zero-bias below 128 (186-358 vs 39-339 its).
- SPICE: 54 runs, all plain Newton, zero fallbacks (fallback logging works: a smoke-test separate 8x8 used dynamic gmin stepping). KLU's numerical pivoting changes fill per circuit: separate 256 seed 2 had 1.56e7 vs 1.26-1.32e7 fill, 34 vs 19-20 s.
- Costs at 1024 (Spingarn, AMD): setup 4.5 s (symbolic 1.0, numeric 2.9), ~540 ms/it = resolvent 150-165 + network 253 (triangular solve 198) + KCL check 112 + update 22; 2.83 GB. SPICE at 512: one-time 21-41 s (load + KLU reorder), 4.1 / 5.5 / 13.1 s per Newton it (memristor / lumped / separate), ~90% KLU factor; 3.1-4.5 GB vs Spingarn 0.72-0.98 GB. cpu/wall ~1.0 everywhere (single-threaded).
- Agreement with SPICE at word/bit nodes: memristor <= 1.4e-10 V, 1D1R <= 2.5e-7 V (separate limited by SPICE's 1 mohm uramp diode).

Diode port sweep (same day, PC, seed 0, `diode_port_sweep.py`, results/diode-port-256 and -512; new rule
`diode_gamma="geometric"` = per-cell sqrt(R_min R_max)): at 256 with zero-bias memristors, scalar ports give 300 ohm cap
(>5000), 1k 2287, 3k 765, 10k 265, 30k 285, 100k 942, 300k 2819, 1M cap; per-cell R_min (the default) 951, per-cell
geometric **166** (best; sector memristors 266, best scalar 278). At 16x16 the order is reversed (R_min 39, geometric 112),
so the best port depends on size. At 512: geometric 470 its / 100 s (sector 496) vs R_min 3029 / 579 s and SPICE 256 s,
i.e. 2.6x faster than SPICE; growth x2.8/doubling remains (lower constant, same slope). It nearly equals lumped 1D1R
with sector matching (166 vs 168 at 256, 470 vs 517 at 512). Untested: per-cell multiples of geometric, 1024
(projected ~20 min).

Confirmed by sweep B (same day, PC, 3 seeds, 16-512, results/network-models-2-dgeometric, `--diode-gamma geometric`,
SPICE and lumped reused from network-models-1): zero-bias memristor + geometric diode 113/124/130/129/170/519 its at
16..512 (lumped 1D1R sector: 115/126/130/129/168/517, i.e. the same at every size), runtime 103 s at 512 vs SPICE 262 s
(2.5x; 2.7x at 256, 1.8x at 128; loses only at 16) and vs the R_min port 576 s; memory 0.98 vs SPICE 4.54 GB. Sector
memristor + geometric diode stays flat ~250 through 256 and joins at 512 (531, 103 s). The geometric port loses to R_min
below 64 (113 vs 39 at 16), ties at 64, wins from 128. Agreement with SPICE <= 2.2e-7 V.
1024, one run (seed 0, results/network-models-2-dgeometric-1024): 1595 its (x3.4 from 470), 1388 s (23 min), 865
ms/it (triangular solve 356, check 207, resolvent 180; lnz 1.73e8 vs lumped 1.17e8), setup 7.5 s, 4.12 GB. Lumped sector at
1024 seed 0: 1594 its, so the two formulations match in iterations 16-1024; separate is ~1.6x dearer per iteration.
Seeds 1-2 at 1024 added later that day (same folder): 1607 / 1597 its, 1170 / 1167 s, 723-726 ms/it, so seed 0's 865
ms/it was ~20% session noise (network solve and check slower, same code); median 1170 s.

Three-model comparison, 2026-10-19 (the user's task in instructions.md: memristor, ideal separate, Shockley separate; fixed
Gamma, AMD, alpha 1/0.8, to 1024): memristor reused from network-models-1, ideal case = sweep B + the 1024 seeds above,
Shockley case in [[shockley-separate-findings]]. CSVs + pgfplots in latex/network_models/. For this task the user dropped
the one-hour rule at 1024 (nothing in it projected over an hour anyway) and declined 768x768; whether the rule stands for
future sweeps is open, so ask.

Instrumentation added that session: Spingarn history profile fields (assembly/symbolic/numeric, resolvent/network/triangular_solve/check/update), SPICE `fallbacks` + `notes_count`, runner `cpu_time`.

**Why:** first PC comparison of the three cell models on one circuit with fixed matching; it shows where fixed-Gamma Spingarn wins (memristor, lumped 1D1R with sector matching to 512) and where it doesn't (separate diodes), and corrects the 2026-10-19 feasibility claim that sector matching keeps lumped 1D1R flat (true only to 256).

**How to apply:** quote "sector flat on lumped 1D1R" only up to 256; for the separate form use diode_gamma="geometric" at scale (best tested from 256 up, not shown optimal); the diode port, not the memristor matching, is the lever there. Compare against [[adaptive-matching-findings]] and [[diode-models-findings]] (laptop) only qualitatively: different machine and, for the memristor, a different circuit than the nominal crossbar. Related: [[diode-1d1r-findings]], [[spingarn-solver-findings]].
