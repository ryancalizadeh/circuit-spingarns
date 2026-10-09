---
name: python-environment
description: "which interpreter runs this project (venv/), what is installed in it (incl. cvxopt for CHOLMOD and its quirks), that ngspice works on the desktop but is deliberately not installed on the laptop, and why scikit-sparse is not an option here"
metadata:
  type: project
---

Run everything with `venv/Scripts/python.exe` (Python 3.14, created by the user 2026-10-06). The system interpreters (3.14 and 3.12 on PATH, plus the Store alias) have no scipy. The venv holds numpy, scipy 1.18, matplotlib 3.11, PySpice, and since 2026-10-08 cvxopt 1.3.3.

On the desktop (2026-10-09, see [[measurement-machine]]) the venv lacked cvxopt, which was installed then (1.3.3), and ngspice *is* available: ngspice-47 at C:\Spice64 plus ngspice.dll in PySpice's Spice64_dll\dll-vs, set up 2026-10-05 per the README; all 33 tests, test_spice.py included, pass there. pytest is not installed; the tests run as plain scripts. The no-SPICE rule below applies to the laptop.

PySpice is installed only because `algorithms/__init__.py` imports `SpiceSolver`, so every import of `algorithms` needs it. The ngspice library is **not** installed, and the user does not want it installed for now (2026-10-06, and again 2026-10-08 when the factorization comparison was run without SPICE): skip SPICE runs and `testing/test_spice.py`.

CHOLMOD comes from cvxopt, because SciPy has no Cholesky and scikit-sparse ships only a source package for Windows (this machine has no compiler, conda or vcpkg). cvxopt's CHOLMOD (SuiteSparse 7.11, 64-bit indices) is compiled without METIS, so its only own ordering is AMD. It bundles its own OpenBLAS 0.3.31 at `cvxopt/.libs/libopenblas.dll`, separate from numpy's and scipy's, which is how `algorithms/spingarns.py` sets CHOLMOD's thread count. Quirks: `cholmod.symbolic(A, p=None)` is rejected (omit `p` instead), and `cvxopt.matrix` rejects int64 arrays whose buffer format is `<q` (arrays built from ctypes); plain numpy int64 (`q`) works.

Node.js is not installed either, so the dataviz palette validator cannot run.

**Why:** finding a working interpreter cost a full search in the 2026-10-06 session, and finding a CHOLMOD binding that installs on Windows/Python 3.14 another in 2026-10-08.

**How to apply:** use the venv directly. If a task needs SPICE results, ask the user before installing ngspice, and use the existing numbers in [[ngspice-baseline-findings]] in the meantime (they come from another machine, see [[measurement-machine]]).
