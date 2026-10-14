# Task: diode models in the 1D1R crossbar (Shockley vs ideal, lumped vs internal node)

You are picking up from a previous session. Read `memory/MEMORY.md` (it's loaded via CLAUDE.md), then these memory files in full before starting: `diode-1d1r-findings`, `adaptive-matching-findings`, `selector-1s1r-findings`, `ngspice-baseline-findings`, `measurement-machine`, `python-environment`, `report-format`, `working-feedback-style`. Then skim `report_nonsmooth.md` §1–2 and `report_adaptive_matching.md` §1, §4.

## Goal

Test how the diode model and the circuit formulation change the Newton (SPICE) vs Spingarn comparison on the 1D1R crossbar, in three cases:

1. **Shockley diode, lumped 1D1R.** Diode and memristor in series, treated as one branch (as the current `1d1r` cell is).
2. **Shockley diode, internal node.** The diode is its own branch from the word node to a new internal node per cell, and the memristor is a second branch from there to the bit node.
3. **Ideal diode, internal node.** As case 2, with the ideal diode instead of the Shockley one.

The user knows that cases 1 and 2 are the same physical circuit (only the formulation differs) and wants the comparison anyway. Deliverable: a short report focused on figures and results (see "Report" below).

## Background (why this test)

- **On the existing ideal-diode 1D1R, SPICE's Newton beat fixed-matching Spingarn from 128×128 up.** The lumped cell i = g_m(max(v, 0)) is single-valued and Lipschitz, with slope in [0, 1/R_min], because the series memristor caps it. So it is a semismooth equation with an always-positive-definite Jacobian A·G·Aᵀ: easy for Newton, which behaves like an active-set method on the kink. Meanwhile a reverse-biased cell reflects +1 under any finite port resistance, so Spingarn's rate bound is 1, and the collective mismatch grows with size (36 → 968 iterations).
- **Adaptive matching (`spingarn_adaptive`) fixes that** (22–31 iterations, flat), but it refactorizes about as often as Newton does.
- **What hurts Newton is not nonsmoothness but a linear model that is wrong over the step taken:**
  - Exponentials: the Shockley conductance rises e-fold per nV_T ≈ 26 mV. Newton overshoots, can overflow, and crawls back by ~nV_T per step. SPICE's `D` element has junction-voltage limiting (pnjlim) for this; behavioural `exp()` sources don't.
  - Set-valued elements: a conducting ideal diode on its own branch is a short (infinite conductance), which can't be stamped.
- **Predictions to test, which the user cares about.** State in the report which held:
  - **H1, case 1:** Newton on the lumped Shockley law converges fast, like the ideal 1D1R, because the series memristor caps the slope and the cell law is "eventually linear" on both sides. Spingarn with fixed matching slows with size as before; adaptive matching restores flat counts. Spingarn gains nothing from the smoother diode.
  - **H2, case 2:**
    - SPICE with a `D` element (limiting) converges, perhaps with more iterations than case 1.
    - SPICE with a behavioural `exp()` diode (no limiting) struggles: more iterations, fallbacks (gmin, source stepping, transient) or failures.
    - Spingarn is robust: each diode's solve is a bounded 1-D monotone root-find, so nothing overshoots or overflows. But it is slow with fixed matching (the diode's resistance spans from ~0 to ∞), and adaptive matching should approach Newton-like refactorization counts.
  - **H3, case 3:**
    - SPICE cannot represent an ideal diode exactly. With a regularized stand-in (`uramp(V)/R_on`, small R_on), the matrix gets ill-conditioned as R_on → 0; watch for Newton failures and accuracy loss.
    - Spingarn solves the exact model, but every diode reflects fully: b = −|a|, i.e. −1 when conducting and +1 when blocking, for any γ. So matching can't help, α < 1 is required, and the rate is probably poor and size-dependent.
- **Correction to an earlier claim:** the Shockley diode's sector is unbounded, not finite. In deep reverse, i → −I_s and r_d = nV_T/(i + I_s) → ∞. Lumped with the memristor the cell's sector is [R_min, ∞), like the ideal 1D1R.

## Implementation

Work in the existing structure; read `Config.py`, `devices.py`, `algorithms/spingarns.py`, `algorithms/spice.py`, `algorithms/ordering.py`, `testing/test_spingarn.py`, `testing/test_config.py` first.

**Device laws (devices.py, Config.py)**
- Shockley law: i = I_s·expm1(v/(n·V_T)), with parameters I_s, n, V_T (default V_T = 25.85 mV) stored in `Config.params()`. Use `expm1` for accuracy near 0.
- Case 1 lumped cell (e.g. `device="1d1r_shockley"`): the series combination has no closed form with the tanh memristor. It needs a nested monotone solve like `devices.series_solve` for 1S1R: for port resistance γ, solve x_d + x_m + γ·i = a with i = Shockley(x_d) = g_m(x_m), reusing `tanh_root` for g_m⁻¹. γ = 0 gives the cell law itself.
  - Bracket: |x_d| ≤ |a|.
  - **Pitfall:** under adaptive matching γ reaches 1e8 Ω, so `a` can be ~1e5 V. Never evaluate `exp` at a bracket end naively. Work in current or log space, or clip the bracket by the largest current the memristor allows.
- Shockley branch alone (case 2): its resolvent has a closed form with no overflow, through the Wright omega function (`scipy.special.wrightomega`, which solves ω + ln ω = z). With c = n·V_T:
  - x + γ·I_s·(e^{x/c} − 1) = a gives x = a + γ·I_s − c·ω(L), where L = (a + γ·I_s)/c + ln(γ·I_s/c).
  - Checked numerically: it never overflows, even at a = ±1e5 V and γ = 1e8, and |x| ≤ |a|. At large positive a it loses digits to cancellation (relative residual up to ~4e-10), and one Newton polishing step brings every case tested to ≤ 4e-15. Check the residual in a test.
- Ideal diode branch alone (case 3): v = min(a, 0), i = max(a, 0)/γ.
- Cell sectors: lumped Shockley [R_min, ∞); Shockley branch [0, ∞); ideal diode branch {0, ∞}. A port-resistance rule must never return 0, so give diode branches an explicit rule and a positive floor.

**Internal-node topology (cases 2, 3)**
- Add it as a `Config` option (e.g. `internal_node=True`): pq extra nodes (3pq in all) and pq extra branches (4pq).
- Many places use `slices[BranchKind.DEVICE]`, so keep the blast radius small. One option is to keep the DEVICE block as the memristor branches, add a DIODE block, and make solvers that don't support it (Condat-Vu, Scellier) raise a clear error.
- Update `incidence`, `current`, `conductance`, `params` and the SPICE node naming.
- **Ordering:** eliminate the internal nodes first, then use the existing nested-dissection order of the word and bit nodes. Each internal node has degree 2, so eliminating it creates exactly the word–bit coupling of the lumped matrix, adding no fill beyond the lumped case. Verify with `lnz`.

**Spingarn (algorithms/spingarns.py)**
- `cell_resolvent` and the per-branch port resistances must handle the new blocks.
- Extend adaptive re-matching (`_Rematching`) to the diode branches, clipped to [floor, open_resistance].
- Keep every existing solver **bit-identical**. Before editing, record iterations plus a hash of v and i for `spingarn`, `spingarn_sector` and `spingarn_slu` on memristor, 1D1R and 1S1R circuits (24×20, seed 5), and diff afterwards.

**Newton baseline (needed for case 1)**
- SPICE can only run the internal-node netlist, so "Newton on the lumped law" has to be in-repo. Add a registered Newton solver on nodal KCL: Jacobian A·G·Aᵀ from `cfg.conductance`, factorized with the same CHOLMOD and nested-dissection code (`factorize`, then `solve.refactor` each step), and the same KCL convergence test (`algorithms.common.kcl_check`).
  - Default to plain undamped Newton, like SPICE.
  - Add an optional step-halving safeguard. A plain line-Newton 2-cycled at a 1D1R kink in the Scellier work.
  - Record iterations and the number of halvings.
- It runs on any formulation (lumped or internal node), which also gives the "Newton on our own factorization" comparison that `report_adaptive_matching.md` §4 asks for. `testing/test_config.solve` is a minimal reference version (SuperLU, no safeguard).

**SPICE (algorithms/spice.py)**
- Case 2: two variants, selectable by a solver option:
  - a `D` element from w to m (`.model ... D(IS=… N=… RS=0)`; RS = 0, or ngspice adds its own internal node), with the memristor B-source from m to b, as the 1S1R netlist already does;
  - a behavioural source `I = Is*(exp(V(w,m)/(N*Vt))-1)`. Check whether ngspice clamps `exp()` in B-sources, and record what you find.
- Case 3: `I = uramp(V(w,m))/Ron` with an R_on sweep (e.g. 1 Ω, 1e-3 Ω, 1e-6 Ω).
- Note: gmin adds conductance across semiconductor junctions; note how that affects reverse currents at the 1 pA scale.
- Always check `history["method"]` (newton, gmin stepping, source stepping, transient) and trust convergence only after the KCL check: the transient-op fallback reports success falsely.

**Tests**
- Add tests in the style of `testing/test_spingarn.py`:
  - independent restatements of each law;
  - resolvent residuals over wide input ranges, including huge γ;
  - agreement of Spingarn, the Newton solver and SPICE on small circuits of each case;
  - lumped vs internal-node solutions agreeing at the word and bit nodes;
  - topology counts.
- Run all seven test files as plain scripts at the end (there is no pytest).

## Experiments

**Circuit:** the 1D1R circuit of `report_nonsmooth.md` §2:
- E ∈ [−1, 1.5] V;
- R_min 1–10 kΩ and R_max 50–200 kΩ per cell;
- wires 5 Ω, sources 50 Ω, loads 1 kΩ.

**Diode parameters:** choose them so that the diodes are actually exercised.
- With I_s = 1e-14 A (silicon pn) the turn-on is ~0.6 V, and at scale conducting cells in this circuit sit at millivolts (ideal-diode median 0.009 V at 256×256), so nearly everything would be subthreshold.
- Try a Schottky- or selector-like I_s (e.g. 1e-9 to 1e-6 A, n = 1–1.5) alongside 1e-14.
- Report operating-point diagnostics per size: the fraction of diodes reverse-biased, subthreshold (r_d > R_min), in the knee, and on (r_d < R_min). Also report the output change against the ideal-diode 1D1R.
- If no parameter set exercises the knee at scale, say so rather than pressing on.

**Solvers per case** (adjust names to what you register):

| case | Newton (in-repo) | SPICE | Spingarn |
|---|---|---|---|
| 1 lumped Shockley | plain, and with safeguard | — (can't express the lumped law) | fixed (α 0.8), adaptive (α 2/3), split |
| 2 internal-node Shockley | plain, and with safeguard | `D` element, behavioural `exp` | fixed, adaptive |
| 3 internal-node ideal | — (no finite Jacobian) | `uramp/R_on`, R_on sweep | fixed, α ∈ {0.5, 2/3, 0.8}; try a few diode port resistances |

**Phases**
1. **Feasibility, in-process:** 1×1 to 32×32, 3 seeds, like `selector_feasibility.py`. Check agreement, iteration counts, SPICE methods and failures, and the operating-point regimes. Pick the I_s and settings to scale.
2. **Scaling, runner:** sizes up to 256×256 (the laptop cap; the PC can go further), 3 seeds, isolated runs, with a timeout (e.g. 1800 s) and iteration budgets. Leave out configurations that clearly diverge or blow up in phase 1, and say so.

**Measurement rules (from memory)**
- Name the machine (`hostname`: `ryan-work` is the laptop, DESKTOP-EFQTE92 the PC).
- Time only on AC with "Best performance": check `(Get-CimInstance Win32_Battery).BatteryStatus` (2 = AC) and ask the user to plug in if it is 1.
- Never run anything else CPU-heavy during a timed sweep.
- Compare runtimes only within one machine and one sweep.
- Ask the user before any single run or sweep expected to take more than ~30 minutes.

**Accuracy:** compare solutions at the shared word and bit nodes (max |Δv|). Report the recorded KCL residuals per solver: absolute ones on 1D1R, because the relative ones are meaningless on dead lines. Don't project Spingarn's v onto KVL by least squares (`report_adaptive_matching.md`, accuracy notes).

## Report

`report_diode_models.md` at the repo root, figures in `report_assets/diode_models/` drawn by `diode_models_figures.py` from the result directories, using `plotting.py`'s palette (run the dataviz palette validator for any new colour pairing; see `python-environment`). Run scripts that print Greek letters with `python -X utf8`.

The user wants the **figures and results** to carry the report, with a **very brief theory overview**: about 5–8 bullets drawn from "Background", no derivations. Suggested structure:
1. Overview: the theory bullets, plus 3–5 headline results.
2. Setup: one short table (circuit, diode parameters, solvers, sizes, seeds, machine).
3. Results, one subsection per case, each with its figure(s) and a compact numbers table. Candidates:
   - cell I–V curves (ideal vs Shockley, linear and log current);
   - iterations vs size;
   - runtime vs size;
   - SPICE method and failure breakdown per size;
   - operating-point regime fractions vs size;
   - one residual-history plot for a representative large case;
   - lumped vs internal node, side by side.
4. Which of H1–H3 held, as bullets.
5. Reproduction commands and a files-changed table.

Write it for the user (the expert who owns the project), in the style of the existing reports: bullets, numbers with ranges over seeds, the machine named, and limitations stated.

## Working style

- **Narrate as you go:** what you're checking, problems, fixes and measured outcomes (see `working-feedback-style`). Surface deviations from H1–H3 or from theory immediately, and let the user decide design calls.
- **Use `venv/Scripts/python.exe`.** Keep scratch scripts in your scratchpad; anything that produces reported numbers belongs in the repo and writes to `results/`.
- **Don't commit** unless asked.
- **At the end:** add a memory note `memory/diode-models-findings.md` (with an index line in `memory/MEMORY.md`), and update related notes that your results make stale (e.g. `todo-sinh-devices`, `diode-1d1r-findings`). Notes go in the repo's `memory/` folder, not a machine-local directory.
