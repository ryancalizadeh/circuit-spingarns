---
name: todo-sinh-devices
description: "deferred by the user (2026-10-09): explore sinh-like (exponential) memristor I-V, i ~ sinh(beta v), as a model where SPICE's Newton may struggle; out of scope for now"
metadata:
  type: project
---

The user asked (2026-10-09) to note sinh-like devices for later; they are out of scope for now.

Idea: a strongly exponential memristor read law, i ~ G sinh(beta v). This is closer to measured oxide RRAM read currents than the tanh law, which bends the wrong way (sublinear). Why it may separate the solvers:
- Newton overshoots on exponentials. A behavioral source gets no junction-voltage limiting (unlike SPICE's built-in diode), so large beta may force gmin or source stepping.
- For Spingarn the law is monotone. Its sector is unbounded globally (incremental resistance -> 0), but the no-gain property bounds every device voltage by max |E|, so over the operating range the sector is [R(E_max), R(0)], giving a size-independent bound. However K ~ cosh(beta E_max) grows fast with beta, so the constant may be large.

**Why:** after the 1S1R check ([[selector-1s1r-findings]]) showed that bounded-slope kinks don't hurt Newton, exponential growth is the remaining physical candidate for a Newton disadvantage.

**How to apply:** when the user returns to device models, start with a small feasibility check: SPICE iterations and fallbacks vs beta, alongside Spingarn's iterations and bound. Confirm with the user before running anything larger.
