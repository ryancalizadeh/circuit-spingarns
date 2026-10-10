"""
DC operating point of the crossbar from ngspice, driven through PySpice's shared-library interface.

The netlist is written as text rather than through PySpice's Circuit API: at 256x256 the Circuit API spends ~3 s of
pure Python building element objects against ~0.1 s for text, and that overhead would be timed as SPICE runtime.
Node w{r}_{c} / b{r}_{c} is Config's word / bit node (r, c), and s{r} is the internal node between source E[r] and
its series resistance. Each memristor is a behavioral current source carrying eq. (11), with its own cell's R_min and
R_max; a 1D1R cell applies it to max(v, 0) through ngspice's uramp. A 1S1R cell is two behavioral sources in series
through an internal node m{r}_{c}: the selector of devices.py from w{r}_{c} to m{r}_{c}, and the memristor on to b{r}_{c}.
"""
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError, NgSpiceShared

from algorithms.common import Solver, SolveOutput, kcl_check

# PySpice logs every ngspice stderr line that is not a "Warning:" as an error, including routine notes such as
# "Starting dynamic gmin stepping", and warns that ngspice 47 is unsupported although the parts of the shared-library
# API used here are unchanged. The solver reads ngspice's output itself and records it in the run history instead.
logging.getLogger("PySpice").setLevel(logging.CRITICAL)

_ngspice = None


def ngspice():
    """The process-wide ngspice instance; the shared library holds a single simulator per process."""
    global _ngspice
    if _ngspice is None:
        # cffi does not search the DLL's own folder for its dependencies (libomp140, sndfile, samplerate).
        dll_dir = Path(NgSpiceShared.LIBRARY_PATH).parent # type: ignore
        if hasattr(os, "add_dll_directory") and dll_dir.is_absolute() and dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
        _ngspice = NgSpiceShared.new_instance()
    return _ngspice


def _ramp(x, delta):
    """ngspice expression for devices._ramp of the expression x: max(x, 0), rounded quadratically over |x| < delta."""
    if delta == 0:
        return f"uramp({x})"
    d = repr(float(delta))
    return f"(uramp({x}+{d})*uramp({x}+{d})-uramp({x}-{d})*uramp({x}-{d}))/{4 * float(delta)!r}"


def netlist(cfg, options=()):
    """SPICE netlist of the crossbar described by `cfg`, with the given `.options` entries and an `.op` analysis."""
    p, q = cfg.p, cfg.q
    g_lin = (1 / cfg.R_max_cell).reshape(p, q)
    g_tanh = (1 / cfg.R_min_cell - 1 / cfg.R_max_cell).reshape(p, q)
    # the memristor's voltage: across the cell, beyond the selector in a 1S1R cell, through ngspice's uramp in a 1D1R
    x = {"memristor": "V(w{r}_{c},b{r}_{c})", "1d1r": "uramp(V(w{r}_{c},b{r}_{c}))",
         "1s1r": "V(m{r}_{c},b{r}_{c})"}[cfg.device]
    tail = "m" if cfg.device == "1s1r" else "w"  # the memristor's word-side node
    R_wire, R_source, R_load = repr(float(cfg.R_wire)), repr(float(cfg.R_source)), repr(float(cfg.R_load))
    lines = [f".title crossbar {p}x{q} seed={cfg.seed}"]
    lines += [f"V{r} s{r} 0 {float(cfg.E[r])!r}" for r in range(p)]
    lines += [f"RS{r} s{r} w{r}_0 {R_source}" for r in range(p)]
    lines += [f"RW{r}_{c} w{r}_{c} w{r}_{c + 1} {R_wire}" for r in range(p) for c in range(q - 1)]
    lines += [f"RB{r}_{c} b{r}_{c} b{r + 1}_{c} {R_wire}" for r in range(p - 1) for c in range(q)]
    lines += [f"B{r}_{c} {tail}{r}_{c} b{r}_{c} I={float(g_lin[r, c])!r}*{xrc}+{float(g_tanh[r, c])!r}*tanh({xrc})"
              for r in range(p) for c in range(q) for xrc in [x.format(r=r, c=c)]]
    if cfg.device == "1s1r":
        vs = "V(w{r}_{c},m{r}_{c})"
        V_th = repr(float(cfg.V_th))
        selector = (f"I={1 / float(cfg.R_off)!r}*{vs}+{1 / float(cfg.R_on) - 1 / float(cfg.R_off)!r}*"
                    f"({_ramp(f'{vs}-{V_th}', cfg.delta)}-{_ramp(f'-{vs}-{V_th}', cfg.delta)})")
        lines += [f"BS{r}_{c} w{r}_{c} m{r}_{c} " + selector.format(r=r, c=c) for r in range(p) for c in range(q)]
    lines += [f"RL{c} b{p - 1}_{c} 0 {R_load}" for c in range(q)]
    if options:
        lines.append(".options " + " ".join(options))
    lines += [".op", ".end"]
    return "\n".join(lines)


def node_names(cfg):
    """SPICE names of Config's nodes, in Config's node order: word nodes, then bit nodes, each row-major."""
    p, q = cfg.p, cfg.q
    return [f"w{r}_{c}" for r in range(p) for c in range(q)] + [f"b{r}_{c}" for r in range(p) for c in range(q)]


# ngspice's notes when a fallback for the operating point succeeds; plain Newton prints none. When every fallback fails,
# ngspice still leaves an op plot behind (holding the last iterate), so total failure needs a marker of its own.
_OP_METHODS = [("operating point could not be simulated", "failed"),
               ("transient op finished successfully", "transient op"),
               ("source stepping completed", "source stepping"),
               ("gmin stepping completed", "gmin stepping")]


def op_method(notes):
    """How ngspice reached its operating point, from its stderr notes; "failed" if it did not."""
    text = "\n".join(notes).lower()
    for marker, method in _OP_METHODS:
        if marker in text:
            return method
    return "newton"


# ngspice `rusage everything` fields recorded in the run history.
_RUSAGE = {
    "Total iterations": "spice_iterations",
    "Circuit Equations": "equations",
    "Circuit original non-zeroes": "nonzeros",
    "Circuit fill-in non-zeroes": "fill_in",
    "Total analysis time (seconds)": "spice_analysis_time",
    "Matrix load time": "matrix_load_time",
    "Matrix reorder time": "matrix_reorder_time",
    "Matrix factor time": "matrix_factor_time",
    "Matrix solve time": "matrix_solve_time",
    "Current ngspice program size": "process_memory_mb",
}


def rusage(ng):
    # After a failed analysis, rusage still prints its statistics but also an internal error on stderr
    # ("if_getstat: ... can't get a name for analysis parameter 53"), which PySpice raises as a command error.
    try:
        lines = ng.exec_command("rusage everything", join_lines=False)
    except NgSpiceCommandError:
        lines = ng.stdout.splitlines()
    stats = {}
    for line in lines:
        key, sep, value = line.partition("=")
        match = re.match(r"\s*([-+0-9.eE]+)", value)
        if sep and match and key.strip() in _RUSAGE:
            stats[_RUSAGE[key.strip()]] = float(match.group(1))
    return stats


def read_op(ng, plot, path):
    """Node voltages of operating-point `plot`, by node name, via a binary rawfile (far faster than per-vector API
    calls on large circuits)."""
    ng.exec_command(f"setplot {plot}")
    ng.exec_command("set filetype=binary")
    ng.exec_command(f"write '{Path(path).as_posix()}'")  # ngspice keeps double quotes as part of the file name
    data = Path(path).read_bytes()
    k = data.index(b"Binary:")
    header = data[:k].decode().splitlines()
    n = int(next(line for line in header if line.startswith("No. Variables:")).split(":")[1])
    start = header.index("Variables:") + 1
    names = [line.split()[1] for line in header[start:start + n]]
    values = np.frombuffer(data, dtype=np.float64, count=n, offset=data.index(b"\n", k) + 1)
    return {name[2:-1] if name.startswith("v(") else name: value for name, value in zip(names, values)}


class SpiceSolver(Solver):
    """
    DC operating point (.op) of the crossbar from ngspice, through PySpice.

    tol is ngspice's RELTOL and max_iterations its ITL1, the Newton iterations allowed before ngspice falls back to
    gmin stepping, source stepping and finally a transient ramp (so total iterations can exceed it). Both default to
    SPICE's standard values; ngspice raises any ITL1 below 100 to 100. matrix_solver selects KLU (default) or
    ngspice's older SPARSE 1.3, whose ordering
    becomes the bottleneck beyond ~64x64. Any other keyword is passed to ngspice as an option, e.g. abstol=1e-15 or
    gminsteps=0 (True for a bare flag).

    A run counts as converged only if ngspice produces an operating point and that point satisfies KCL to ngspice's
    own tolerance (see kcl_check): the transient-ramp fallback reports success even for circuits with no DC solution.
    """
    name = "spice"
    default_tol = 1e-3
    default_max_iterations = 100

    def __init__(self, tol=None, max_iterations=None, matrix_solver="klu", **options):
        if matrix_solver not in ("klu", "sparse"):
            raise ValueError(f"matrix_solver must be 'klu' or 'sparse', not {matrix_solver!r}")
        super().__init__(tol, max_iterations, matrix_solver=matrix_solver, **options)
        ngspice()  # load and initialize the shared library here, outside the timed solve calls

    def describe(self):
        return dict(name=self.name, reltol=self.tol, itl1=self.max_iterations,
                    itl1_effective=max(100, self.max_iterations), **self.options)

    def spice_options(self):
        extra = {k: v for k, v in self.options.items() if k != "matrix_solver"}
        return [f"reltol={float(self.tol)!r}", f"itl1={int(self.max_iterations)}", self.options["matrix_solver"]] + [
            k if v is True else f"{k}={v}" for k, v in extra.items()]

    def solve(self, cfg):
        ng = ngspice()
        history = {}
        raw = Path(tempfile.gettempdir()) / f"crossbar_op_{os.getpid()}.raw"
        t = time.perf_counter()
        text = netlist(cfg, self.spice_options())
        history["netlist_time"] = time.perf_counter() - t
        before = set(ng.plot_names)
        try:
            t = time.perf_counter()
            ng.load_circuit(text)
            history["load_time"] = time.perf_counter() - t

            t = time.perf_counter()
            try:
                ng.exec_command("run")
            except NgSpiceCommandError:
                pass  # raised for any note on stderr; the outcome is judged from the plots and KCL below
            history["run_time"] = time.perf_counter() - t
            notes = ng.stderr.splitlines()
            solver_lines = [line for line in ng.stdout.splitlines() if "Direct Linear Solver" in line]
            history["linear_solver"] = solver_lines[0] if solver_lines else None
            history["notes"] = notes[:20]
            history.update(rusage(ng))
            iterations = int(history.get("spice_iterations", 0))

            plots = [name for name in ng.plot_names if name.startswith("op") and name not in before]
            if not plots:
                history["method"] = None
                return SolveOutput(None, None, False, iterations, history)
            history["method"] = op_method(notes)
            if history["method"] == "failed":
                return SolveOutput(None, None, False, iterations, history)

            t = time.perf_counter()
            by_name = read_op(ng, plots[0], raw)
            u = np.array([by_name[name] for name in node_names(cfg)])
            history["extract_time"] = time.perf_counter() - t

            t = time.perf_counter()
            abstol = float(self.options.get("abstol", 1e-12))
            v, i, ok, history["kcl_residual"], history["kcl_residual_rel"] = kcl_check(cfg, u, self.tol, abstol)
            history["verify_time"] = time.perf_counter() - t
            return SolveOutput(v, i, ok, iterations, history)
        finally:
            for command in ("destroy all", "remcirc"):
                try:
                    ng.exec_command(command)
                except NgSpiceCommandError:
                    pass
            raw.unlink(missing_ok=True)
