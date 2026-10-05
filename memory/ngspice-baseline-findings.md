---
name: ngspice-baseline-findings
description: "measured ngspice-47 behaviour on the crossbar that decides how SPICE results must be read: false-success fallback, ITL1 floor, SPARSE vs KLU scaling, flat Newton count, memory at scale"
metadata:
  type: project
---

Measured 2026-10-05 with ngspice-47 driven by PySpice 1.5, nominal crossbar parameters:

- ngspice's last `.op` fallback, the transient op, reports "finished successfully" even for a circuit with no DC solution. SPICE convergence is therefore only trusted after the KCL check in `algorithms/spice.py`.
- ngspice silently raises any ITL1 below 100 to 100.
- ngspice's default matrix solver is SPARSE 1.3, ~n^2.2 from reordering (64x64: 1.7 s; 256x256 unfinished after 10 min). KLU is ~n^1.4-1.8 (256x256: 2.5 s run; 512x512: 30 s run, 41 s total, 2.8 GB). The solver defaults to KLU; this choice was put to the user.
- Newton took 4 iterations at every size from 1x1 to 512x512; SPICE's scaling cost is LU fill-in (20x the original non-zeros at 512x512), not iteration count.
- Projected 1024x1024: ~6-7 min and 12-15 GB. 2048x2048 likely exceeds the 32 GB machine.
- Out of memory (tested with 512x512 under a 1.5 GB per-process job limit): ngspice dies during netlist loading with exit code 0xC00000FD (stack overflow), not with an error message. Runs are process-isolated since 2026-10-05, so this is recorded as status "crashed" and the sweep continues.

User decisions (2026-10-05): runtime is the total solve time, harness overhead included, because large sizes are what matter and overhead is negligible there. SPICE stays at RELTOL 1e-3; the user will tune tolerances later.

**Why:** these decide whether a SPICE failure in a sweep is a convergence failure, a memory limit or a configuration artifact, which is the core of [[project-goals]].

**How to apply:** when reading or reporting SPICE results, check `history["method"]` and `kcl_residual_rel`, state which matrix solver was used, and treat failures at very large sizes as possible memory exhaustion before calling them non-convergence.
