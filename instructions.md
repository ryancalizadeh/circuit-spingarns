# Task for the next agent: SPICE vs Spingarn on three crossbar models, up to 1024×1024

Written 2026-10-19 by the agent that ran the network-models sweeps that day, as a handoff. Read all of it before
starting; §4 lists decisions to confirm with the user first.

**Status: done, 2026-10-19 evening (PC).** The user's decisions: I_s 1e-9, n 1; reuse case 2 (1024 seeds 1-2 added);
no one-hour rule; no 768; SPICE's Shockley diode as a behavioural exp source (`spice_exp`), not the D element, to 256 ×
3 seeds plus one 512 run. Data: results/network-models-3-shockley (+ network-models-1, -2-dgeometric, -2-dgeometric-1024).
CSVs and pgfplots code: latex/network_models/ (regenerate with `python network_models_latex.py`). Findings:
memory/shockley-separate-findings.md, memory/network-models-findings.md.

## 1. The task (from the user)

Compare SPICE against Spingarn on three cases:

1. **Memristor alone**, Spingarn with zero-bias matching and with sector matching. *The user said data from previous
   runs may be reused here* (see §3).
2. **1D1R, ideal diode, diode and memristor on separate branches** (internal node). Spingarn: zero-bias matching on the
   memristor, geometric matching on the diode.
3. **1D1R, Shockley diode, separate branches.** Spingarn: zero-bias memristor, geometric diode.

Rules:
- No advanced tricks: **fixed Γ (no adaptive re-matching)** and **no geometric nested dissection (GNDO)**, i.e.
  CHOLMOD's own ordering (`ordering="builtin"`, AMD in cvxopt's build).
- **alpha = 0.8** for both 1D1R cases, **alpha = 1** for the memristor alone.
- Go **up to 1024×1024** for whichever methods can handle it.
- When the data is collected, **save it to CSV** loadable from a LaTeX project, and **write pgfplots LaTeX code** for the
  plots. Compile it and look at the rendered result before handing it over.

The user wants running narration while you work: what you are checking, what broke, what you tried, and the numbers as
they come in (memory/working-feedback-style.md). Name the machine with every runtime or memory figure.

## 2. Settings that map the task onto the code

| task wording | code |
|---|---|
| Spingarn, zero-bias memristor | `SpingarnSolver(gamma="r_lo")` (γ = R_min per cell) |
| Spingarn, sector memristor | `SpingarnSolver(gamma="geometric")` (γ = √(R_min·R_max) per cell) |
| geometric diode port | `diode_gamma="geometric"` (γ_d = √(R_min·R_max) per cell; rule added 2026-10-19) |
| no GNDO | `ordering="builtin"` |
| no adaptive | plain `SpingarnSolver`, not `SpingarnAdaptive*` |
| alpha | `alpha=0.8` (1D1R), `alpha=1.0` (memristor) |
| SPICE | `SpiceSolver()` defaults: ngspice-47, KLU, RELTOL 1e-3, ITL1 100 |
| ideal separate 1D1R | `Config(..., device="1d1r", internal_node=True)`; SPICE writes the diode as `uramp(V)/1 mΩ` |
| Shockley separate 1D1R | `Config(..., device="1d1r_shockley", internal_node=True, I_s=..., n=...)` |

Keep everything else as in the 2026-10-19 sweeps so the data stays comparable: the **1D1R circuit for all cases**
(`E_range=(-1, 1.5)`, per-cell `R_min` log-uniform in 1–10 kΩ, `R_max` in 50–200 kΩ, wires 5 Ω, sources 50 Ω, loads
1 kΩ; `network_models_experiment.CIRCUIT`), Spingarn tol 1e-9 (its KCL test is SPICE's own), `blas_threads=1`, 3 seeds
(0–2), runs isolated in their own processes.

Use `network_models_experiment.py` (sweep / summarize / export). It already does isolation, peak memory, CPU time, a
per-run profile line in `sweep.log`, SPICE fallback logging, resume, skip-after-failure, the 1024 runtime-limit rule
(`--runtime-limit-at 1024=3600`), solution agreement (`--reference-dir`), and CSV export. To add the Shockley case you
must extend three module-level dicts: `CASES` (e.g. `"1d1r_shockley_separate": dict(CIRCUIT, device="1d1r_shockley",
internal_node=True, I_s=..., n=...)`), `ALPHA` (0.8) and `CSV_CASES` (a short column name, e.g. `"shockley"`; export
raises KeyError without it). `--diode-gamma geometric` applies to every case in one invocation and appends
`_dgeometric` to the Spingarn names; run the memristor case (if at all) in a separate invocation without it.

## 3. Data that already exists (PC, DESKTOP-EFQTE92, 2026-10-19)

All in `results/` (git-ignored). Every run converged unless noted.

- **`network-models-1/`** — the main sweep: cases memristor, 1d1r_lumped, 1d1r_separate × solvers `spice`,
  `spingarn_zero_bias`, `spingarn_sector` (diode port R_min), all with exactly the settings above (alpha 1 / 0.8,
  builtin ordering, fixed Γ). Sizes 16–512 × 3 seeds for everything, plus 1024 × 3 seeds for the Spingarn runs projected
  under an hour (memristor zero-bias and sector, lumped sector). SPICE was not run at 1024 (KLU limit, below). **The
  memristor case here is exactly case 1 of the task**, so reuse it: SPICE 16–512, both Spingarn variants 16–1024.
  Memristor medians: SPICE 44.4 s at 512 (5 Newton its, 3.09 GB); zero-bias 3.8 s at 512 / 16.3 s at 1024 (20–21 its,
  2.83 GB at 1024); sector 17.2 s / 69.7 s (117–119 its). `runtime_plot.tex` + `runtime.csv` there are a working
  pgfplots template (see §6).
- **`network-models-2-dgeometric/`** — 1d1r_separate with `--diode-gamma geometric`, solvers
  `spingarn_zero_bias_dgeometric` and `spingarn_sector_dgeometric`, 16–512 × 3 seeds. **This is exactly case 2's
  Spingarn data up to 512.** Its SPICE reference is `network-models-1`'s 1d1r_separate SPICE runs (same circuits and
  seeds). Medians, zero-bias + geometric diode: 113/124/130/129/170/519 its at 16..512; 7.4 s at 256, 103 s at 512
  (SPICE 20.0 / 262 s; 0.98 vs 4.54 GB at 512).
- **`network-models-2-dgeometric-1024/`** — case 2 at 1024, **one seed only** (seed 0): 1595 its, 1388 s (23 min),
  4.12 GB, 865 ms/iteration.
- `diode-port-256/`, `diode-port-512/` — the diode-port sweep that chose the geometric port (`diode_port_sweep.py`).
- Nothing exists yet for case 3 (Shockley). Older Shockley data (laptop, 8–128, results/diode-models-*) is on the
  laptop, not this PC, and must not be compared with PC numbers.

Summaries: `summary.md` in each folder; findings in `memory/network-models-findings.md`.

## 4. Decisions to confirm with the user before running

1. **Shockley parameters.** The task doesn't give I_s or n. Earlier work used I_s = 1e-9 and 1e-6 with n = 1 (both put
   30–35% of diodes in the knee at 64–128); Config's default I_s = 1e-14 leaves the diodes mostly subthreshold, which
   barely exercises the exponential. Ask which I_s/n (or both).
2. **Case 2 reuse.** The user allowed reuse explicitly only for the memristor case, but case 2's data exists with the
   exact settings (§3) except seeds 1–2 at 1024. Ask whether to reuse it (then only 1024 seeds 1–2 remain, ~23 min each)
   or rerun case 2 in one fresh sweep with SPICE (~40 min for 16–512, plus ~70 min at 1024).
3. **The one-hour rule.** Earlier the user said to skip any 1024 run that would take over an hour; it is implemented as
   `--runtime-limit-at 1024=3600` (power-law projection from 256 and 512). Ask whether it still applies.
4. **768×768.** Not in the 2026-10-19 sweeps. It is the only size where SPICE might still work on the separate cases
   (see below); offer it, with the cost.

## 5. Known behaviour to expect (do not rediscover)

- **SPICE/KLU size limit**: ngspice's KLU fails at 1024×1024 on any two-node-per-cell crossbar and at 768×768 with an
  internal node per cell (1S1R, 2026-10-09): "singular matrix" at the first Newton step, every fallback fails, factor
  time 0, ~5–6 min, ~10 GB. The user said not to rerun SPICE at 1024 (memory/ngspice-baseline-findings.md). Both
  separate cases have three nodes per cell, so expect SPICE to fail at 768 as well; use `--spice-max-size 512`, or try
  768 with one seed (the sweep skips the remaining seeds after the KLU-limit signature).
- **SPICE fallbacks**: all 54 SPICE runs on 2026-10-19 were plain Newton. The fallback chain is logged per run in
  `history["fallbacks"]`. For Shockley, SPICE's `D` element (the default, `diode="D"`) stalled in plain Newton on ~1 seed
  in 3 at I_s ≤ 1e-9 (laptop, 4–32), rescued by gmin stepping, so expect fallbacks there. Do not use `spice_exp` (exp()
  behavioural diodes): it fell back on every run from 16×16 and took up to 1047 s at 128. The `D` element is not the exact
  Shockley law (V_T = 25.865 mV, gmin across the junction, cubic reverse law); its KCL test uses the as-solved law and
  `kcl_exact_residual` records the exact one.
- **Spingarn iteration growth**: with these fixed matchings iteration counts grow ~×3 per doubling past 256 on 1D1R; case
  2 reached 1595 its at 1024. Case 3 with the geometric diode port is untested; the previous agent's guess (not
  measured) is that it behaves like case 2 at scale and worse than an R_min port below ~64, and that the best port drifts
  towards the knee (smaller) as I_s grows. Shockley separate uses a closed-form Wright-omega resolvent, so its
  per-iteration cost should be close to case 2's (192 ms at 512, 865 ms at 1024).
- **Spingarn 1D1R KCL rel residual ~0.9** in summaries is a known artifact (abstol binds on dead lines), not a bad
  solution; judge accuracy by the agreement table (≤ 2.2e-7 V vs SPICE on ideal 1D1R, limited by SPICE's 1 mΩ diode).
- **Memory**: SPICE uses ~4.6× Spingarn's memory at 512; most of it is ngspice's per-element representation, not the LU
  factor (hypothesis with supporting data in the 2026-10-19 conversation; untested).

## 6. CSV and LaTeX

- `python network_models_experiment.py export <dir>` writes `runtime.csv` (wide: `N`, then `<case>_<solver>` median
  runtime with `_min`, `_max`, `_n`; `nan` where nothing converged) and `runs.csv` (one row per run, skipped included).
  It only covers runtime and one directory. For this task you will need **one CSV per metric** (runtime, iterations, peak
  memory; per-iteration and one-time cost are in each record's `costs` too) merged across folders (memristor from
  `network-models-1`, case 2 from wherever you settle it, case 3 new). A small merge script is the clean way.
- `results/network-models-1/runtime_plot.tex` is a compiled, checked template: standalone + pgfplots groupplots, one
  panel per case, log-log axes, shared legend via `legend to name` + `\pgfplotslegendfromname` (needs **two** pdflatex
  passes). Colours from plotting.py, validated with the dataviz skill's checker: SPICE `#EB6834` (circle marks),
  zero-bias `#2A78D6` (squares), sector `#1BAF7A` (triangles). The teal is below 3:1 contrast on white, so every series
  needs its own marker. Pitfall: `log basis x=2` with minor ticks is a pgfplots error; use the default base with explicit
  `xtick`/`xticklabels`.
- MiKTeX's `pdflatex` and `pdftoppm` are installed on the PC: compile in the scratchpad, render to PNG, look at it.

## 7. Code state and environment

- **Nothing from 2026-10-19 is committed.** Modified: `algorithms/spingarns.py` (profiling fields in the history;
  `diode_gamma="geometric"`), `algorithms/spice.py` (`fallback_stages`, `history["fallbacks"]`, `notes_count`),
  `run_experiments.py` (`cpu_time` in every run's history), `testing/test_spice.py`, `testing/test_spingarn.py`,
  `memory/MEMORY.md`. New: `network_models_experiment.py`, `diode_port_sweep.py`,
  `memory/network-models-findings.md`. Tests run as plain scripts (no pytest): test_spice 15, test_spingarn 26,
  test_experiments 4 passed. Commit only if the user asks.
- Machine: check `hostname`; the PC is `DESKTOP-EFQTE92` (6-core i5-11600KF, 31.8 GB, ~15 GB free). Never compare its
  numbers with the laptop's (`ryan-work`). See memory/measurement-machine.md.
- Python: `venv/Scripts/python.exe` only (memory/python-environment.md).
- Pass `</dev/null` to Python commands in Bash; a command reading stdin (e.g. a stray `cat >` without a heredoc) hangs.
- Background Bash tasks are capped at 2 h. For a longer sweep, launch it detached with PowerShell `Start-Process
  ... -RedirectStandardOutput ... -WindowStyle Hidden -PassThru` and watch `sweep.log` with a background until-loop.
- The sweep's isolated runs use multiprocessing spawn, which **re-imports the running script from disk in every child**:
  never leave a syntax error in `network_models_experiment.py` while a sweep runs (run `python -m py_compile` after
  edits).
- The two long-lived Python processes (PIDs 7588/1944 on 2026-10-19) are VS Code's isort extension, not stray runs.
- Timing noise on the PC: identical work ran 42 ms vs 57–62 ms per iteration in one stretch on 2026-10-19. Compare
  solvers within one sweep (the script interleaves them per size and seed) and rely on iteration counts where runtimes
  disagree.
- Project memory lives in `memory/` (CLAUDE.md): read `memory/MEMORY.md` and the files it links, especially
  network-models-findings, ngspice-baseline-findings, diode-models-findings and measurement-machine. Add findings there,
  one fact per file, with a line in MEMORY.md.
