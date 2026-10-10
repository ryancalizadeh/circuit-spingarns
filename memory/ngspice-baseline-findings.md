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
- ~~Projected 1024x1024: ~6-7 min and 12-15 GB.~~ Wrong: measured 2026-10-09 on the desktop of [[measurement-machine]], **SPICE/KLU fails at 1024x1024** ("singular matrix: check node 0" on the first Newton step, then every gmin/source/transient-op fallback fails; ~290 s, 2 seeds, ngspice at 4.7-8.9 GB with ~12-20 GB RAM free). A purely linear 1024 crossbar (devices -> R_min resistors) fails identically, and KLU's numeric factorization never runs (factor time 0, reorder only 7.4 s vs 61 s at 768), so it is a KLU size limit in ngspice's build, not Newton non-convergence. 768x768 still converges (4 Newton its, ~2.5 min, L+U ~122M entries, 6.3 GB). Unverified guess: a 2^31-byte limit in the 32-bit KLU between ~1.5 GB of LU (768) and ~3 GB (1024). The user said not to rerun SPICE at 1024 (2026-10-09).
- 1S1R crossbars (2026-10-09, PC): SPICE/KLU already fails at 768x768, with the same signature as above (singular matrix at the first Newton step, factor time 0, every fallback fails, ~333 s, ~10 GB peak). The 1S1R matrix has 1.77M equations (an internal node per cell) and ~2x the LU fill per factorization, so the size limit arrives earlier. 512x512 works (8-14 Newton its, 100-146 s, 5.2 GB). See [[selector-1s1r-findings]].
- After a failed analysis, ngspice's `rusage` prints an internal error to stderr, which PySpice raises; and ngspice leaves an op plot behind even on total failure. `algorithms/spice.py` handles both since 2026-10-09 (`method: "failed"`, status not_converged).
- At RELTOL 1e-3 on the desktop (2026-10-09), SPICE's KCL residuals from 64x64 up were 3-6e-10, about the same as Spingarn's at tol 1e-9, not ~1e-11.
- Out of memory (tested with 512x512 under a 1.5 GB per-process job limit): ngspice dies during netlist loading with exit code 0xC00000FD (stack overflow), not with an error message. Runs are process-isolated since 2026-10-05, so this is recorded as status "crashed" and the sweep continues.

User decisions (2026-10-05): runtime is the total solve time, harness overhead included, because large sizes are what matter and overhead is negligible there. SPICE stays at RELTOL 1e-3; the user will tune tolerances later.

**Why:** these decide whether a SPICE failure in a sweep is a convergence failure, a memory limit or a configuration artifact, which is the core of [[project-goals]].

**How to apply:** don't run SPICE at 1024x1024 or larger with KLU (it fails after ~5 min); 768 is the largest size measured to work. When reading or reporting SPICE results, call the 1024 failure a KLU size limit, not a convergence failure; check `history["method"]` and `kcl_residual_rel`, state which matrix solver was used, and treat failures at very large sizes as possible memory exhaustion before calling them non-convergence.
