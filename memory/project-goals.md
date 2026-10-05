---
name: project-goals
description: "compare SPICE and a custom Spingarn partial-inverses algorithm for non-ideal memristive crossbar arrays, with emphasis on scaling and convergence at large sizes"
metadata:
	node_type: memory
	type: project-goals
---

Compare simulations of non-ideal memristive crossbar arrays using SPICE and a custom algorithm based on Spingarn's method of partial inverses.

**Why:** the project investigates how runtime and iteration count scale as array size grows, and whether SPICE fails to converge for extremely large arrays while the custom algorithm continues to succeed. Treat this as a hypothesis to test, not an assumed outcome.

**How to apply:** when proposing experiments or implementing changes, keep both methods comparable on the same array sizes and non-ideal conditions; track runtime, iteration count, and convergence as array size increases.
