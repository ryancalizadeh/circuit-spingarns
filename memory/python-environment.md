---
name: python-environment
description: "which interpreter runs this project (venv/), what is installed in it (incl. cvxopt for CHOLMOD and its quirks), that ngspice-47 works on both machines (laptop since 2026-10-13, DLL only), and why scikit-sparse is not an option on Windows"
metadata:
  type: project
---

Run everything with `venv/Scripts/python.exe` (Python 3.14, created by the user 2026-10-06). The system interpreters (3.14 and 3.12 on PATH, plus the Store alias) have no scipy. The venv holds numpy, scipy 1.18, matplotlib 3.11, PySpice, and since 2026-10-08 cvxopt 1.3.3.

On the desktop (2026-10-09, see [[measurement-machine]]) the venv lacked cvxopt, which was installed then (1.3.3), and ngspice *is* available: ngspice-47 at C:\Spice64 plus ngspice.dll in PySpice's Spice64_dll\dll-vs, set up 2026-10-05 per the README; all 33 tests, test_spice.py included, pass there. pytest is not installed; the tests run as plain scripts. The no-SPICE rule below applies to the laptop.

PySpice is installed only because `algorithms/__init__.py` imports `SpiceSolver`, so every import of `algorithms` needs it. On the laptop the user declined ngspice twice (2026-10-06, 2026-10-08), then chose to install it on 2026-10-13 for the Condat-Vu comparison. It is the DLL only, no C:\Spice64: ngspice-47_dll_64.7z (md5 b6fa6b27...) extracted into PySpice's NgSpice\Spice64_dll, plus samplerate.dll and sndfile.dll from ngspice-47_64.7z's bin. Windows' own `tar` (bsdtar 3.8.8 with liblzma) extracts .7z, so no 7-Zip is needed; its "empty or unreadable filename" warnings are harmless metadata entries. All 9 SPICE tests pass there.

CHOLMOD comes from cvxopt, because SciPy has no Cholesky and scikit-sparse ships only a source package for Windows (the laptop has no compiler, conda or vcpkg; the PC was not checked). cvxopt's CHOLMOD (SuiteSparse 7.11, 64-bit indices) is compiled without METIS, so its only own ordering is AMD. It bundles its own OpenBLAS 0.3.31 at `cvxopt/.libs/libopenblas.dll`, separate from numpy's and scipy's, which is how `algorithms/spingarns.py` sets CHOLMOD's thread count. Quirks: `cholmod.symbolic(A, p=None)` is rejected (omit `p` instead), and `cvxopt.matrix` rejects int64 arrays whose buffer format is `<q` (arrays built from ctypes); plain numpy int64 (`q`) works.

The dataviz skill ships a Python twin of its palette validator (`scripts/validate_palette.py`, found 2026-10-13), which runs on either machine without Node: `venv/Scripts/python.exe -I -X utf8 <skill>/scripts/validate_palette.py "#hex,..." --mode light [--pairs all | --ordinal]` (`-X utf8` because the Windows console can't print its arrows). Node.js is not installed on the laptop. The PC has Node 16.15.1 (checked 2026-10-10), which loads the `.js` version as CommonJS and fails; that one runs when copied, still named `validate_palette.js`, into a scratch folder holding `package.json` = `{"type":"module"}`. Finding with it (2026-10-13): palette slot 6 (green #008300) fails CVD against slot 2 (spice's orange) at Delta E 3.2 (protan), so condat_vu is pinned to slot 7 (violet) in plotting.py.

**Why:** finding a working interpreter cost a full search in the 2026-10-06 session, and finding a CHOLMOD binding that installs on Windows/Python 3.14 another in 2026-10-08.

**How to apply:** use the venv directly. SPICE runs work on both machines now; still compare runtimes only within one machine (see [[measurement-machine]]). If a fresh venv ever lacks ngspice, repeat the DLL recipe above (README "SPICE setup").
