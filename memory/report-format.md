---
name: report-format
description: "user prefers write-ups as Markdown reports in the repo (report_*.md + PNGs from a figure script) over interactive artifact web pages; thesis figures as CSV + pgfplots code (latex/<topic>/), compiled and checked"
metadata:
  type: feedback
---

For explanatory write-ups the user prefers a Markdown report in the repo to an interactive web page/artifact (stated 2026-10-10 when asking for the nested-dissection explainer: "I would prefer an md report, but if it makes more sense as an interactive artifact webpage then just do that").

**Why:** the repo's reports (report.md, report_nonsmooth.md, report_nested_dissection.md) live next to the code they describe and are reproduced by scripts.

**How to apply:** default to `report_<topic>.md` at the repo root, figures in `report_assets/<topic>/`, drawn by a `<topic>_figures.py` script using plotting.py's palette (see [[python-environment]] for running the palette validator). Offer an artifact only when interaction clearly adds something.

For figures meant for the LaTeX thesis the user asked (2026-10-19) for the data as CSV loadable from a LaTeX project plus pgfplots code, compiled and looked at before handing over. Done as latex/network_models/: one wide CSV per metric (N, then per series median, _min, _max, _n; nan where nothing converged), a style file (colours/markers from plotting.py, \NMdata data-dir macro), one fig_*.tex tikzpicture per metric for \input into a figure environment, and preview.tex/.pdf (two pdflatex passes for `legend to name`). The CSVs come from a merge script (network_models_latex.py) that refuses mixed machines and duplicate runs. pgfplots pitfall: groupplot reads `group style` only from its own options, not from inside a style.
