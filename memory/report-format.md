---
name: report-format
description: "user prefers write-ups as Markdown reports in the repo (report_*.md + PNGs from a figure script) over interactive artifact web pages"
metadata:
  type: feedback
---

For explanatory write-ups the user prefers a Markdown report in the repo to an interactive web page/artifact (stated 2026-10-10 when asking for the nested-dissection explainer: "I would prefer an md report, but if it makes more sense as an interactive artifact webpage then just do that").

**Why:** the repo's reports (report.md, report_nonsmooth.md, report_nested_dissection.md) live next to the code they describe and are reproduced by scripts.

**How to apply:** default to `report_<topic>.md` at the repo root, figures in `report_assets/<topic>/`, drawn by a `<topic>_figures.py` script using plotting.py's palette (see [[python-environment]] for running the palette validator). Offer an artifact only when interaction clearly adds something.
