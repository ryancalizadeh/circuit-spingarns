---
name: diode-models-findings
description: "Shockley vs ideal diode, lumped vs internal-node 1D1R (2026-10-14, laptop, 8-128): step-halving Newton on our CHOLMOD is fastest wherever it runs (ties adaptive on ideal lumped, 2-3x faster on Shockley); plain Newton 2-cycles / crawls / breaks Cholesky; SPICE exp fails badly, D element is not exact Shockley; adaptive Spingarn works on exact ideal internal diodes (22-27 flat)"
metadata:
  type: project
---

Measured 2026-10-14 on the laptop (AC, Best performance; see [[measurement-machine]]). Report: report_diode_models.md; results/diode-models-feasibility (1-32, in-process, 4 diode sets), diode-models-scale-Is1e-9 (all 4 cases), diode-models-scale-Is1e-6 (Shockley cases); sizes 8-128 (user's choice for this run), 3 seeds. 89 tests pass (19 new).

Code: Config `device="1d1r_shockley"` (I_s, n, V_T) and `internal_node=True` (DIODE block after LOAD, 3pq nodes, 4pq branches; ideal internal diodes: current() uses the cell-total convention, conductance() raises); `algorithms/newton.py` (`newton`, `newton_safeguarded`: CHOLMOD+ND, fixed-pattern Jacobian, kcl_check); SPICE `diode="D"|"exp"`, `ron`, `spice_exp`; `Config.line_potentials` for word/bit comparisons. Internal nodes ordered first: lnz = lumped lnz + 3pq exactly.

- **The Newton baseline [[adaptive-matching-findings]] asked for, at 128x128**: step-halving Newton 0.20-0.33 s (7-12 its) vs adaptive Spingarn 0.26-0.82 s vs SPICE 1.2-2.3 s. Tie on ideal lumped 1D1R (split 1.2x faster), Newton 2.1-3x faster on Shockley cells. So adaptive's lead over SPICE is the linear solver and ngspice overhead, not the iteration. Not measured above 128.
- **Plain (SPICE-style, undamped) Newton on our factorization is unusable**: 2-cycles at the ideal 1D1R kink from 64x64 (step stuck at 2.0-2.3 V; SPICE's own Newton doesn't, cause unknown); on Shockley internal nodes it crawls at n V_T per step (40-42 its) or overshoots to ~1e14 S conductances and CHOLMOD reports a non-positive pivot (which one is rounding luck). Step halving fixes all.
- **SPICE `D` element** converges (13-17 its), but stalls in plain Newton on ~1 seed in 3 at I_s <= 1e-9 from 4x4-32x32 (gmin stepping rescues; ITL1 3000 doesn't). It is not exact Shockley: V_T 25.865 mV, gmin 1e-12 S per junction, SPICE3 cubic reverse law below -3 n V_T (up to 0.4% of I_s off) -> 1.3e-4 V shift at I_s 1e-6, 128x128 (first-order prediction matches). KCL for SPICE is checked against the as-solved law (`as_solved_current`).
- **SPICE behavioural exp()**: ngspice clamps exp at 1e99 (flat above arg 227.96). Fallback in every run from 16x16; 27-1047 s at 128 (KLU re-ordering up to 1039 s), up to 1.1 GB; one false convergence at 4e71 V that kcl_check passed (inf <= RELTOL*inf) until it was fixed to require finite residuals.
- **Spingarn**: lumped and internal formulations take the same iterations (fixed 340-376, adaptive 45-47 at 128); internal is cheaper per iteration (closed-form Wright-omega resolvent). Exact ideal internal diodes: adaptive 22-27 flat (same as lumped), 5x faster than SPICE; fixed needs diode port ~R_min (100 ohm or 100 kohm: 900-3000 its). Split adaptive fails on Shockley cells and ideal internal diodes (unopened strongly reflecting branches at alpha 1).
- **Operating point**: knee (r_d within 10x of R_min) exercised at every size: 30-35% of diodes at 64-128 for I_s 1e-9/1e-6, none "on"; I_s 1e-14 mostly subthreshold. Output change vs ideal 1D1R: ~50% (1e-14), 20-26% (1e-9), 10-12% (1e-6).

**Why:** this settles the deciding experiment of report_adaptive_matching §4 (against attributing the speedup to Spingarn, up to 128) and shows exponential devices hurt unlimited Newton, not Newton with a step-length safeguard.

**How to apply:** compare Spingarn against `newton_safeguarded`, not plain `newton` or SPICE alone, before claiming algorithmic wins. Check SPICE `D` runs against the as-solved law and quote the exact-law gap. Open questions: why SPICE's limited Newton stalls (a pnjlim replica in our Newton would test it), and whether Spingarn's cheap iterations win at 512-1024 on the PC. Related: [[diode-1d1r-findings]], [[ngspice-baseline-findings]], [[todo-sinh-devices]].
