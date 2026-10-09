---
name: prior-work
description: "closest prior art to the Spingarn crossbar solver: the wave-digital Scattering Iterative Method (Bernardini et al. 2018) is essentially the same algorithm; also Minty, Spingarn/Douglas-Rachford, Chaffey & Sepulchre; what may still be new"
metadata:
  type: reference
---

Checked 2026-10-08 (read the SIM paper in full, plus searches):

- **SIM, essentially our algorithm:** A. Bernardini, P. Maffezzoni, L. Daniel, A. Sarti, "Wave-based analysis of large nonlinear photovoltaic arrays", IEEE Trans. Circuits Syst. I, 2018 (re.public.polimi.it/handle/11311/1061617). Same waves a = v + Zi, b = v - Zi with diagonal free port resistances; same scattering matrix S = 2Q^T (QGQ^T)^-1 QG - I (cut-set form of our S_Gamma); local 1-D Newton per element then b = 2v - a; global step by one Cholesky of QGQ^T per operating point; theorem rho(prod S D) <= max reflection |R - Z|/(R + Z) for positive incremental resistances (proof via the orthogonal G^1/2 S Z^1/2, as in report §1.3); "adaptation" Z = incremental resistance for speed. Up to 6,400 PV units, 33-35x faster than Spectre. Unrelaxed (our alpha = 1). No mention of Douglas-Rachford, Spingarn or monotone operators. Follow-ups use SIM for virtual-analog audio (Proverbio/Bernardini/Sarti EUSIPCO 2020; WD Newton-Raphson, TASLP 2021; dynamic scattering-matrix recomputation).
- **Optimization side:** Spingarn's partial inverses (1983) = Douglas-Rachford (Lions-Mercier 1979) on a subspace; alpha = 1 is Peaceman-Rachford; the branch map a -> b is the Cayley transform of the branch law.
- **Monotone circuit theory:** Minty, "Monotone networks" (1960) and an algorithm for monotone networks (1961); Chaffey & Sepulchre, "Splitting algorithms and circuit analysis" (arXiv 2208.04765) and "Circuit analysis using monotone+skew splitting" (arXiv 2211.14010, Condat-Vu, periodic steady state); Shahhosseini, Chaffey, Sepulchre, "Operator-splitting methods for neuromorphic circuit simulation" (arXiv 2505.22363, 2025).
- **Crossbar solvers** found so far use node-based or preconditioned Krylov/stationary methods (e.g. arXiv 2109.07929); no application of SIM or partial inverses to memristive crossbars was found.

What may still be new here: memristive crossbars; ~10^6 nonlinear devices (SIM: 6,400); the explicit SIM = Spingarn/Peaceman-Rachford identification with relaxation alpha; the study of matching rules for a sector-bounded device (R_min vs geometric vs operating range, size-independence 1x1-1024x1024); geometric nested dissection + CHOLMOD for the global step.

**Why:** the README calls impedance matching as preconditioning "the main contribution", and SIM's adaptation already contains that idea; novelty claims must be positioned against SIM.

**How to apply:** cite Bernardini et al. 2018 (and Minty, Spingarn, Lions-Mercier) whenever the method is described or claimed; frame contributions as application, scale, analysis and the optimization-wave-digital bridge, not the iteration itself. The user's notes mention an "Alberto" and Tom Chaffey; whether that Alberto is Alberto Bernardini is unconfirmed.
