---
name: measurement-machine
description: "the two machines runtime/memory numbers come from (8-core Lunar Lake laptop, 15.5 GB; 6-core i5 PC, 31.8 GB), which report sections each one measured, and their threading and noise quirks"
metadata:
  type: project
---

The user moves between two machines. Always name the machine ("the laptop", "the PC"), never "this machine": which one a session runs on changes from day to day. The PC's hostname is DESKTOP-EFQTE92, so `hostname` tells them apart.

**The laptop** (used since 2026-10-06; report.md §6): Intel Core Ultra 7 256V, 4 performance + 4 low-power efficiency cores (8 threads), 15.5 GB RAM, of which only ~4 GB was available with VS Code and Slack open (2026-10-08), and a 21 GB pagefile. Hostname `ryan-work`. ngspice-47 (DLL in the venv) installed 2026-10-13, see [[python-environment]]; the user caps laptop simulations at 256x256 (2026-10-13).

The numbers in report.md §2–3 and in [[ngspice-baseline-findings]] (2026-10-05) were not measured on the laptop, so runtimes and memory must not be compared with the laptop's. Re-measured on the laptop 2026-10-08: SuperLU at 1024x1024 took 32.7–34.4 s total (§2–3: 32 s).

Threading: multithreaded OpenBLAS slows CHOLMOD on the laptop's CPU. With all 8 threads the factorization ran 2–10x slower than with 1, and 4 threads was also slower than 1 or 2, most likely because work lands on the slow efficiency cores and the fast cores wait for them. So the CHOLMOD solvers default to `blas_threads=1` (user decision 2026-10-08, see [[spingarn-solver-findings]]).

Power (2026-10-13): **on battery the laptop throttles sustained single-thread work ~3x within ~10 s** (Condat-Vu at 128x128: 2.7 ms/iteration for the first second, 7.2-8.4 ms from ~9 s on; Balanced plan, 53% charge). Plugged in with Windows' "Best performance" mode (power overlay GUID ded574b5-...) it holds ~2.6-3.5 ms with occasional ~5 ms windows. This penalizes long runs (minutes) far more than sub-second ones, so it biases solver comparisons. Before any timed run, check `(Get-CimInstance Win32_Battery).BatteryStatus` (2 = AC) and ask the user to plug in if it is 1. Even on AC, back-to-back runs vary: SPICE at 256x256 took 4.9-11.5 s across seeds when run right after a 70-80 s Condat-Vu run.

Noise: on the laptop, single in-process timings varied up to ~2x under memory pressure (the same CHOLMOD/AMD factorization at 1024x1024: 6.3 s in one process, 10.9 s in another), while runner sweeps (isolated process per run, solvers interleaved per seed, 3 seeds) agreed to a few percent.

**The PC** (used since 2026-10-09; report.md §7): desktop DESKTOP-EFQTE92, Intel i5-11600KF, 6 identical cores / 12 threads, 31.8 GB RAM, 12.8 GB pagefile, ~12-17 GB available with Firefox/Discord/VS Code open. ngspice-47 works on the PC (see [[python-environment]]); the 2026-10-05 SPICE numbers match it (512x512: 39-41 s both days), so the PC is very likely the "32 GB machine" of report §3. On the PC multithreaded BLAS does *not* slow CHOLMOD: AMD numeric factorization at 1024 took 3.09 s on 1 thread, 2.39 s on 6, 2.47 s on 12 (1 thread still fastest at 256), which supports the efficiency-core explanation for the laptop. Spingarn runtimes on the PC are ~10% below the laptop's.

**Why:** a runtime or memory figure is meaningless without its machine, and both the threading default and the "don't trust a single timing" rule come from the laptop's hardware.

**How to apply:** check which machine the session is on before measuring or before calling something "not installed"; quote runtimes with their machine; compare solvers only within one machine and one sweep; base performance claims on runner sweeps with several seeds, not on one-off in-process timings; re-check the threading default before trusting it on other hardware.
