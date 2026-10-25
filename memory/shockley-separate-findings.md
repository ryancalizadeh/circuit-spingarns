---
name: shockley-separate-findings
description: "Shockley diode on its own branch (I_s 1e-9, n 1), PC 2026-10-19, 16-1024: fixed-Gamma Spingarn (zero-bias memristor + geometric diode port) takes the same iterations as the ideal diode (494 at 512, 1616 at 1024); SPICE behavioural exp needs gmin stepping always, 37x slower at 256, 25x at 512 (one run, 47 min, 25 GB)"
metadata:
  type: project
---

Measured 2026-10-19 on the PC (see [[measurement-machine]]), `network_models_experiment.py` case `1d1r_shockley_separate` (I_s 1e-9 A, n 1, the user's choice; the 1D1R circuit of [[network-models-findings]]), results/network-models-3-shockley. Spingarn: `spingarn_zero_bias_dgeometric` (gamma R_min on memristors, diode port sqrt(R_min R_max), alpha 0.8, fixed Gamma, CHOLMOD AMD). SPICE: `spice_exp` (behavioural exp, the user's choice over the D element, see [[spice-shockley-exp]]). Medians over 3 seeds unless noted; everything converged.

- **Spingarn iterations equal the ideal-diode case at every size**: 109/122/114/107/160/494/1616 at 16..1024 vs ideal 113/124/130/129/170/519/1597. So with the geometric port the diode law doesn't matter; growth x3/doubling from 256 as on the ideal case. Runtime 8.1 s at 256, 111 s at 512, 1429 s at 1024 (ideal 1170 s); per iteration ~880 ms at 1024, the Wright-omega resolvent 323 ms vs 154 ms for the ideal diode (network solve and check equal). Memory 0.98 GB at 512, 4.12 GB at 1024.
- **spice_exp falls back to dynamic gmin stepping on every run** (78-159 Newton its, 88-96 from 128 up); 0.056/0.19/1.46/23.2/299 s at 16..256, 2808 s at 512 (seed 0 only: the user allowed one run with a 1 h cap and a paging watchdog, no second). KLU reorder dominates the one-time cost (65-176 s at 256, 1150 s at 512); per Newton step 1.6-2.5 s at 256, 17.1 s at 512 (factor), fill 1.3-1.5e7 at 256 vs the D element's 9.6e6.
- **SPICE/Spingarn runtime ratio** 1.4/2.4/4.1/21/37/25 at 16..512. On the ideal diode (uramp) SPICE is only 2.5x slower at 512.
- **spice_exp memory** 1.08/6.16/24.98 GB at 128/256/512 (D element 1.06 GB at 256). At 512 the peak is transient, inside KLU's reorder: 33.4 GB commit and 461 MB RAM left for one 15 s sample, then 2.8 GB resident / 7.0 GB commit for the Newton phase (results/network-models-3-shockley/watchdog_512.log).
- **Agreement** Spingarn vs spice_exp (same exact law): <= 3.8e-10 V to 64, 7.3e-8 at 256, 2.8e-7 at 512; 7.8e-6 at 128 comes from spice_exp seed 0 stopping gmin stepping loosely (KCL rel 3.3e-5, passes RELTOL 1e-3), not from Spingarn.
- D element runs (16-256, 3 seeds; solver "spice" in the same folder, left out of the CSVs at the user's request): plain Newton 14-18 its except gmin stepping on seed 0 at 16/32; 16 s and 1.06 GB at 256, i.e. ~19x faster than spice_exp; 1.3e-7 V from Spingarn at 128 (not the exact law).

Deliverable: was latex/network_models/; since 2026-10-25 that folder holds the 10-seed rerun instead ([[network-models-10seeds-findings]]), which reproduces these numbers.

**Why:** completes the user's three-model SPICE vs Spingarn comparison (memristor / ideal / Shockley, separate branches) to 1024.

**How to apply:** cite the 37x/25x as against SPICE *without junction limiting*; the D element would cut SPICE's time ~19x at 256. The 512 SPICE point is one seed. Related: [[diode-models-findings]] (laptop, where spice_exp behaved the same way at 8-128), [[ngspice-baseline-findings]].
