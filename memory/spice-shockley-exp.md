---
name: spice-shockley-exp
description: "user's choice (2026-10-19): in SPICE comparisons, write Shockley diodes as behavioural exp() sources (spice_exp), not ngspice's D element; D-element data is left out of deliverables"
metadata:
  type: feedback
---

On 2026-10-19, mid-sweep, the user said: "don't use SPICE's D element, use a behavioural exp source instead", and then chose to leave the D-element runs already collected out of the CSVs and figures entirely. No reason was given.

**Why:** the D element is not the exact Shockley law (V_T 25.865 mV, gmin across each junction, cubic reverse law; see [[diode-models-findings]]) and it has junction limiting that Spingarn's comparison doesn't get; the behavioural source solves the same law as Spingarn. (My reading, not the user's words.)

On 2026-10-24, with a 1600 s cap per run and 10 seeds, the user chose to run spice_exp at 512x512 on seed 0 only and skip seeds 1-9 once it timed out (it did; [[network-models-10seeds-findings]]), over running all ten to the cap (~4.4 h, each with a ~20-27 GB transient memory peak).

**How to apply:** for Shockley cases use `SpiceExpSolver` / `spice_exp` (network_models_experiment `--algorithms spice_exp`). Warn about its cost up front: it falls back to gmin stepping on every run and grows ~13x per doubling with a transient KLU-reorder memory peak (25 GB at 512 on the PC; [[shockley-separate-findings]]). If the D element is worth showing as a contrast, ask first rather than adding it.
