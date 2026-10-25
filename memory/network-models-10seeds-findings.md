---
name: network-models-10seeds-findings
description: "10-seed fresh rerun of the three-model SPICE vs fixed-Gamma Spingarn task (PC, 2026-10-24/25, 16-1024, 1600 s runtime cap): memristor 12x, ideal 1D1R 3.1x, Shockley 1D1R 35x at 256 and SPICE timed out at 512; Spingarn converged on all 210 runs, every 1024 run under the cap; now the latex/network_models deliverable"
metadata:
  type: project
---

Rerun of instructions.md's task on the PC (see [[measurement-machine]]), 2026-10-24 23:03 to 2026-10-25 09:25, with the user's changes: **no reused data, 10 seeds (0-9) per size, case 1 Spingarn zero-bias only, every run capped at 1600 s of solve time** (killed and recorded as status timeout, "failed to converge within the time limit"). Everything else as [[network-models-findings]] (fixed Gamma, AMD, alpha 1 / 0.8, geometric diode port on 1D1R, SPICE to 512 because of the KLU limit). Data: results/network-models-10seeds/{1-memristor,2-ideal-separate,3-shockley-separate} (runs.jsonl, summary.md, sweep.log; memlog.csv = system memory every 30 s; driver.py + queue.json ran the three sweeps back to back; harness.diff = the uncommitted harness at the time).

Medians over 10 seeds (PC):
- **Memristor**: SPICE plain Newton, 5 its everywhere, 44.2 s / 3.09 GB at 512. Spingarn 19-22 its flat to 1024, 3.69 s at 512 (12.0x), 15.9 s (15.5-16.3) / 2.83 GB at 1024. Faster at every size (3.9x at 16). Agreement <= 1.2e-10 V.
- **Ideal diode, separate**: SPICE 8-16 Newton its (1 of 60 runs needed gmin stepping, 16x16 seed 9), 279 s (235-366) / 4.59 GB at 512. Spingarn 113-130 its to 128, then 170 / 496 / 1596; 0.9x at 16, 3.1x at 256 and 512 (90.6 s, 0.98 GB); 1169 s (1144-1233) at 1024, all under the cap. Agreement <= 2.2e-7 V.
- **Shockley diode, separate** (I_s 1e-9, n 1; SPICE = spice_exp): SPICE gmin stepping on every run, medians 78-94 its; 284 s (166-405) / 4.57 GB at 256 = 35x Spingarn's 8.16 s. 128x128 seed 7: ngspice said gmin stepping completed (1255 its, 135 s) but the KCL check failed (rel 2.6e-3 > RELTOL) -> not_converged. 512 seed 0 timed out at the cap; seeds 1-9 skipped (the user's choice). Spingarn 158 / 486 / 1610 its at 256/512/1024 (same as ideal); 110 s at 512 (>= 14.6x), 1445 s (1418-1479) at 1024, all under the cap. Agreement <= 7.5e-8 V except where SPICE stopped loosely within RELTOL (6.6e-7 V at 16 seed 4, KCL rel 1.9e-5; 7.8e-6 V at 128 seed 0, 3.3e-5).
- Seeds 0-2 are the same circuits as 2026-10-19 and reproduce it closely (memristor SPICE 44.2 vs 44.4 s; Shockley 128 seed 0 agreement 7.8e-6 both times).

Deliverable (overwrote the 3-seed one, the user's permission): latex/network_models/ from `python network_models_latex.py`: CSV per metric with censored medians (a timeout counts as above the cap; nan once half time out) and _runs/_timeout/_capped columns, outcomes.csv, table_runtime.tex / table_iterations.tex (booktabs), five fig_*.tex with min-max bars, a 1600 s cap line, a "timed out" cross and a lower-bound triangle on the speedup; preview.tex/.pdf with setup and results text, compiled and checked.

Harness added for this (run_experiments.run_isolated `runtime_cap`, network_models_experiment `--runtime-cap`, `--skip-seeds-after-timeout [SOLVER...]`): the child sends "solve started" just before the timed solve and the cap counts from there, so start-up, warm-up and circuit generation are not charged. A timeout skips larger sizes only, not the other seeds at that size, unless the solver is named in --skip-seeds-after-timeout.

**Why:** the user's thesis comparison with fresh data, more seeds and a fixed time budget.

**How to apply:** cite these 10-seed numbers rather than the 3-seed ones of [[network-models-findings]] / [[shockley-separate-findings]] (same conclusions, tighter). Spingarn Shockley at 1024 sits at ~1445 s, ~10% under the 1600 s cap: a busier machine could push runs over it. Related: [[ngspice-baseline-findings]], [[spice-shockley-exp]].
