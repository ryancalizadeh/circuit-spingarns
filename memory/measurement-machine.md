---
name: measurement-machine
description: "hardware the current runtime/memory numbers come from (8-core Lunar Lake laptop, 15.5 GB), why the 2026-10-05 numbers are not comparable with it, and its threading and noise quirks"
metadata:
  type: project
---

Since 2026-10-06 the project runs on an Intel Core Ultra 7 256V laptop: 4 performance + 4 low-power efficiency cores (8 threads), 15.5 GB RAM, of which only ~4 GB was available with VS Code and Slack open (2026-10-08), and a 21 GB pagefile.

The numbers in report.md §2–3 and in [[ngspice-baseline-findings]] (2026-10-05) were measured on a different, 32 GB machine, so runtimes and memory from the two machines must not be compared directly. Re-measured here 2026-10-08: SuperLU at 1024x1024 took 32.7–34.4 s total (old machine: 32 s).

Threading: multithreaded OpenBLAS slows CHOLMOD on this CPU. With all 8 threads the factorization ran 2–10x slower than with 1, and 4 threads was also slower than 1 or 2, most likely because work lands on the slow efficiency cores and the fast cores wait for them. So the CHOLMOD solvers default to `blas_threads=1` (user decision 2026-10-08, see [[spingarn-solver-findings]]).

Noise: single in-process timings varied up to ~2x under memory pressure (the same CHOLMOD/AMD factorization at 1024x1024: 6.3 s in one process, 10.9 s in another), while runner sweeps (isolated process per run, solvers interleaved per seed, 3 seeds) agreed to a few percent.

Second machine, used since 2026-10-09 (report.md §7): desktop DESKTOP-EFQTE92, Intel i5-11600KF, 6 identical cores / 12 threads, 31.8 GB RAM, 12.8 GB pagefile, ~12-17 GB available with Firefox/Discord/VS Code open. ngspice-47 works here (see [[python-environment]]); the 2026-10-05 SPICE numbers match it (512x512: 39-41 s both days), so it is very likely the "32 GB machine" of report §3. Here multithreaded BLAS does *not* slow CHOLMOD: AMD numeric factorization at 1024 took 3.09 s on 1 thread, 2.39 s on 6, 2.47 s on 12 (1 thread still fastest at 256), which supports the efficiency-core explanation for the laptop. Spingarn runtimes are ~10% below the laptop's.

**Why:** a runtime or memory figure is meaningless without its machine, and both the threading default and the "don't trust a single timing" rule come from this hardware.

**How to apply:** quote runtimes with their machine; compare solvers only within one machine and one sweep; base performance claims on runner sweeps with several seeds, not on one-off in-process timings; re-check the threading default before trusting it on other hardware.
