---
name: condat-vu-findings
description: "Condat-Vu (Chaffey et al. 2023, Algorithm 1) on the tanh crossbar: iterations ~17 ||M|| with ||M|| ~ 0.90(n+1/2) in closed form, rate exactly 1/(1 + tau R_min) above the Chambolle-Pock balance, ~230x slower than Spingarn at 256 (default crossbar) but fastest up to 64 on a high-R one; tree, rho and stopping-rule findings"
metadata:
  type: project
---

Measured 2026-10-13 on the laptop (on AC, Best performance; see [[measurement-machine]]), solver `condat_vu` (algorithms/condat_vu.py), report_condat_vu.md. Default crossbar (R_min 1k, R_max 100k, E 0.1-0.5 V), tol 1e-9 KCL test as for the other solvers.

- Formulation: line tree (wires, sources, loads; cells are the links). M = -A_T^-1 A_L applied by cumulative sums along lines. ||M||^2 = 1/(4 sin^2(pi/(4p+2))) + 1/(4 sin^2(pi/(4q+2))) exactly (M^T M is a Kronecker sum of [min(j,k)] matrices): ~0.90(n+1/2); smallest singular value ~0.71 (`lines_norm(p, q, smallest=True)`), so M's gains spread ~1.27n.
- Iterations with Chambolle-Pock's balance rho = sqrt(R_min R_load) = 1 kohm: 136 / 258 / 502 / 989 / 1985 / ~3940 at 8..256, ~17 ||M||. Spingarn 5 (r_lo) / ~91 (sector), SPICE 4 Newton, all flat. Runtime at 256 (means of 3 seeds): CV 76 s (61 s at best rho), SPICE 7.6 s, Spingarn 0.33 s (sector 3.6 s). Per iteration CV 19 ms vs Spingarn 38 ms; the shared KCL test is ~40% of a CV iteration.
- Theory: for rho >= balance the linearized rate equals 1/(1 + tau R_min) to 3-4 digits. The slow mode is 100% on the cells with M-coupling ~ 1/spread, i.e. weakly coupled cell patterns get only their own resolvent's damping. Chambolle-Pock Thm 3 (JMIV 2011 sec 5.2) bounds a squared distance by omega^N, so it guarantees only sqrt(omega).
- Rule found: the CP bound is tight on the side whose weakest element is isolated and at its weakest. Default crossbar: cells sit at zero bias (R_min) and some are weakly coupled, so tight for rho >= balance; loads are never isolated (a column's p cells out-conduct its load), so loose below, where the slow mode is a strongly coupled line-sum pattern. High-R crossbar: mirror image (loads isolated, cells saturated).
- Best rho drifts up with size: 400 (8-32), 500 (64), 700 (128, 256); gain over auto ~2x at 16-32 shrinking to 1.24x at 256. The paper's tau = sigma (rho = 1 ohm) is unusable here.
- Tree "devices" (cells in the tree) is 2-6x slower at 2-8 and widening; the CP constant sqrt(mu_R mu_G) favours a minimum-resistance spanning tree.
- Nonlinearity doesn't move CV's count (258/259/260 at 16x16 for E 0.1-0.5/0.5-1.5/1-3 V); the default crossbar is nearly linear at scale (tanh changes outputs <0.05% at 64x64 even at 1-3 V).
- High-R crossbar (R_min 100k, R_max 10M, E 1-3 V; tanh changes outputs 37-89%): mu_R mu_G = 100 gives CV a 10x better constant (22 / 103 / 362 its at 8 / 64 / 256). Runner: CV is the fastest of all four solvers at 8-64 (0.065 s at 64 vs sector 0.096, SPICE 0.22); Spingarn overtakes between 64 and 128, SPICE between 128 and 256 (256: CV 6.35 s, SPICE 5.53, sector 3.23, r_lo 2.63). Auto rho is near-optimal there. Spingarn r_lo's count falls with size there (~300 -> ~70) as larger arrays pull cells out of saturation.
- The paper's stopping test (relative step <= 1e-9) stops with a KCL residual 15x (8x8) to 1200x (256x256) above target: the 1/(1 - rate) factor, growing with ||M||.

**Why:** this is the comparison the notes asked for ("compare against Tom Chaffey's results"); the explicit network step is the structural reason CV's count grows, so it frames every later CV/Spingarn comparison.

**How to apply:** quote CV iteration counts with rho, tree and step product; expect counts ~ ||M|| / sqrt(mu_R mu_G) on any circuit. For the non-smooth step (next), predict CV's 1D1R count stays ~17 ||M||: reverse cells are absorbed by the ReLU-clipped link resolvent and the impedance stays R_min-strongly monotone. That is more robust than Spingarn there, not faster. Confirmed 2026-10-13 ([[scellier-findings]]): 1D1R supported (link current clipped at 0), 111 / 915 / 3523 its at 8 / 64 / 256, ~15||M||, slowest but always converging; 1s1r still raises. Related: [[spingarn-solver-findings]], [[diode-1d1r-findings]], [[prior-work]].
