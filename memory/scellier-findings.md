---
name: scellier-findings
description: "Scellier-style exact coordinate descent (algorithms/scellier.py) on the 1D1R crossbar: node-by-node needs 10^4-10^5 sweeps (unusable), word/bit line blocks ~n^1.4 sweeps and tie Spingarn; SPICE fastest at 256; the line-Newton kink cycle and its fix"
metadata:
  type: project
---

Preliminary sweep 2026-10-13 on the laptop (on AC; power mode not confirmed as Best performance, see [[measurement-machine]]), runner, 3 seeds, 1D1R circuit of [[diode-1d1r-findings]] (E -1..1.5 V, R_min 1-10k, R_max 50-200k), tol 1e-9. Spingarn alpha 0.8, Condat-Vu extended to 1D1R (link current clipped at 0). results/scellier-1d1r (merged from -small 8-64 with all five, -large 128-256 without node CD); no report written, the user asked for plots only.

- Solvers: `scellier` = checkerboard node blocks (the paper's even/odd update), node update = the cell's resolvent with gamma = 1/G_node; `scellier_lines` = all word lines then all bit lines, each a tridiagonal convex problem solved by Newton. Diode kept inside the cell law (as a constraint between free nodes, CD stalls at u = 0).
- Sweeps (mean): node CD 16.9k / 27k / 49k / 110k at 8-64 (2.8 / 6.4 / 26 / 123 s): ~600x slower than everything at 64; stiff 5-ohm wire chains + loads shared by whole lines. Lines 9 / 14 / 25 / 60 / 152 / 436 at 8-256: ~2.5-2.9x per doubling at the large end, like Spingarn's 1D1R growth (36 -> 968).
- Runtime at 256 (s): SPICE 13.4 (12 Newton), lines 38.4, Spingarn 39.4, Condat-Vu 76.4 (3523 its, ~15||M||, slightly below its memristor count: the 1D1R prediction held). Lines fastest at 8-128 (128: 1.39 vs SPICE 1.39, Spingarn 1.94). Per iteration at 256: lines 88 ms, Spingarn 41, CV 22. Memory at 256: SPICE 0.88 GB, others 0.13-0.21.
- Accuracy not matched: iterative solvers stop at the 1 pA abstol (max |KCL| ~1e-12 A), SPICE ~1e-15 A.
- Plain line Newton cycled (2-cycle at a 1D1R kink, 128x128 seed 0); fixed by halving the step on lines where a cell crosses the kink until phi'(t) <= 0.

**Why:** answers whether the Scellier (2024) algorithm is competitive on the non-ideal crossbar; see [[prior-work]] for the paper.

**How to apply:** don't quote node-by-node CD as a contender beyond small arrays; the line-block version is the fair Scellier-family entry and needs no tuning or factorization. Before claims: a full report, matched accuracy, the PC, larger sizes. Plot colours: both share teal slot 2 (node CD dashed); slot 2 is also spingarn_sector's.
