"""
DC operating point of the crossbar from ngspice, driven through PySpice's shared-library interface.

The netlist is written as text rather than through PySpice's Circuit API: at 256x256 the Circuit API spends ~3 s of
pure Python building element objects against ~0.1 s for text, and that overhead would be timed as SPICE runtime.
Node w{r}_{c} / b{r}_{c} is Config's word / bit node (r, c), and s{r} is the internal node between source E[r] and
its series resistance. Each memristor is a behavioral current source carrying eq. (11), with its own cell's R_min and
R_max; a 1D1R cell applies it to max(v, 0) through ngspice's uramp. A 1S1R cell is two behavioral sources in series
through an internal node m{r}_{c}: the selector of devices.py from w{r}_{c} to m{r}_{c}, and the memristor on to b{r}_{c}.

A 1D1R crossbar with internal nodes (Config's internal_node) is netlisted the same way, with the diode from w{r}_{c} to
m{r}_{c} written by the solver's `diode` option:
    "D"    (Shockley diodes) ngspice's own diode element, D(IS, N, RS=0), which has junction-voltage limiting (pnjlim)
           in its Newton iteration. It is not exactly the Shockley law (measured on ngspice-47, 2026-10-14): its thermal
           voltage is k T / q at 27 C, 25.865 mV, so N is scaled to give N V_T = n V_T of the Config; below -3 n V_T it
           takes SPICE3's reverse-bias approximation -I_s (1 + (3 n V_T / (e v))^3), up to 0.4% of I_s off the
           exponential; and ngspice puts gmin (1e-12 S by default) across every junction. spice_diode_current is that law.
    "exp"  (Shockley diodes) a behavioral source I = I_s (exp(V / (n V_T)) - 1): the exact law, without limiting. ngspice
           clamps exp() in behavioral sources at 1e99, i.e. at an argument of ln(1e99) = 227.96, flat beyond (measured on
           ngspice-47 2026-10-14), so it cannot overflow, but its slope vanishes there.
    ideal diodes are regularized to I = uramp(V) / ron, an on-resistance of ron ohms: SPICE cannot represent a short.
A lumped Shockley 1D1R cell has no closed-form law to write, so SPICE takes such circuits only with internal nodes.
"""
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError, NgSpiceShared

from Config import BranchKind
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


# ngspice's thermal voltage k T / q at its default temperature of 27 C, with the constants of its const.h (CODATA 2014)
VT_NGSPICE = 1.38064852e-23 * 300.15 / 1.6021766208e-19
DIODE_MODELS = ("D", "exp")


def spice_diode_current(v, I_s, c, gmin):
    """
    The law of ngspice's diode element with IS = I_s, N V_T = c and RS = 0, as measured on ngspice-47 (2026-10-14; it is
    SPICE3's dioload.c): I_s (e^{v / c} - 1) for v >= -3 c, -I_s (1 + (3 c / (e v))^3) below, plus gmin v.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return np.where(v >= -3 * c, I_s * np.expm1(v / c), -I_s * (1 + (3 * c / (np.e * v)) ** 3)) + gmin * v


def netlist(cfg, options=(), diode="D", ron=1e-3):
    """SPICE netlist of the crossbar described by `cfg`, with the given `.options` entries and an `.op` analysis. diode
    and ron choose how diodes on branches of their own are written (see the module docstring)."""
    if cfg.device == "1d1r_shockley" and not cfg.internal_node:
        raise ValueError("SPICE cannot express the lumped Shockley 1D1R law; use internal_node=True")
    if diode not in DIODE_MODELS:
        raise ValueError(f"diode must be one of {DIODE_MODELS}, not {diode!r}")
    p, q = cfg.p, cfg.q
    g_lin = (1 / cfg.R_max_cell).reshape(p, q)
    g_tanh = (1 / cfg.R_min_cell - 1 / cfg.R_max_cell).reshape(p, q)
    tail = "m" if cfg.device == "1s1r" or cfg.internal_node else "w"  # the memristor's word-side node
    # the memristor's voltage: from its word-side node, through ngspice's uramp in a lumped 1D1R cell
    x = f"V({tail}{{r}}_{{c}},b{{r}}_{{c}})"
    if cfg.device_law == "1d1r":
        x = f"uramp({x})"
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
    if cfg.diode_law == "shockley" and diode == "D":
        lines.append(f".model dshockley D(IS={float(cfg.I_s)!r} N={cfg.n * cfg.V_T / VT_NGSPICE!r} RS=0)")
        lines += [f"D{r}_{c} w{r}_{c} m{r}_{c} dshockley" for r in range(p) for c in range(q)]
    elif cfg.diode_law == "shockley":
        I_s, c_ = repr(float(cfg.I_s)), repr(float(cfg.n * cfg.V_T))
        lines += [f"BD{r}_{c} w{r}_{c} m{r}_{c} I={I_s}*(exp(V(w{r}_{c},m{r}_{c})/{c_})-1)"
                  for r in range(p) for c in range(q)]
    elif cfg.diode_law == "ideal":
        g_on = repr(1 / float(ron))
        lines += [f"BD{r}_{c} w{r}_{c} m{r}_{c} I={g_on}*uramp(V(w{r}_{c},m{r}_{c}))" for r in range(p) for c in range(q)]
    lines += [f"RL{c} b{p - 1}_{c} 0 {R_load}" for c in range(q)]
    if options:
        lines.append(".options " + " ".join(options))
    lines += [".op", ".end"]
    return "\n".join(lines)


def node_names(cfg):
    """SPICE names of Config's nodes, in Config's node order: word nodes, then bit nodes, then any internal nodes, each
    row-major."""
    p, q = cfg.p, cfg.q
    names = [f"w{r}_{c}" for r in range(p) for c in range(q)] + [f"b{r}_{c}" for r in range(p) for c in range(q)]
    if cfg.internal_node:
        names += [f"m{r}_{c}" for r in range(p) for c in range(q)]
    return names


def as_solved_current(cfg, diode, ron, gmin):
    """
    The branch laws of the circuit ngspice actually solves, as a current(v, x0) function for kcl_check, or None where
    they are Config's own: its diode element's law (spice_diode_current) for diode="D", and the on-resistance ron for
    ideal diodes, both on branches of their own. The behavioral exp is the exact Shockley law.
    """
    if cfg.diode_law is None or cfg.diode_law == "shockley" and diode == "exp":
        return None
    d, e = cfg.slices[BranchKind.DEVICE], cfg.slices[BranchKind.DIODE]

    def current(v, x0=None):
        i = cfg.current(v, x0)
        if cfg.diode_law == "ideal":
            i[d] = cfg.device_current(v[d])
            i[e] = np.maximum(v[e], 0.0) / ron
        else:
            i[e] = spice_diode_current(v[e], cfg.I_s, cfg.n * cfg.V_T, gmin)
        return i

    return current


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


def fallback_stages(notes):
    """
    The fallbacks ngspice tried for the operating point, in order, from its stderr notes: a list of [stage, outcome]
    with stage as ngspice names it, lowercased ("dynamic gmin stepping", "true gmin stepping", "source stepping",
    "transient op", ...) and outcome "completed", "failed" or "started" (no outcome noted). Empty for plain Newton.
    """
    stages = []
    for line in notes:
        text = re.sub(r"^(note|warning|error):\s*", "", line.strip().lower())
        start = re.match(r"starting (.+?)\s*$", text) or re.match(r"(transient op) started", text)
        if start:
            stages.append([start.group(1), "started"])
            continue
        end = re.match(r"(.+?) (completed|failed|finished successfully)", text)
        if end:
            stage, outcome = end.group(1), "failed" if end.group(2) == "failed" else "completed"
            match = next((s for s in reversed(stages) if s[0] == stage and s[1] == "started"), None)
            if match:
                match[1] = outcome
            else:
                stages.append([stage, outcome])
    return stages


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
    becomes the bottleneck beyond ~64x64. diode ("D" or "exp") and ron write diodes on branches of their own (see the
    module docstring); they do not matter otherwise. Any other keyword is passed to ngspice as an option, e.g.
    abstol=1e-15 or gminsteps=0 (True for a bare flag).

    A run counts as converged only if ngspice produces an operating point and that point satisfies KCL to ngspice's
    own tolerance (see kcl_check): the transient-ramp fallback reports success even for circuits with no DC solution.
    The test uses the laws ngspice was given (as_solved_current); where those differ from Config's (the diode element,
    the regularized ideal diode), the residual under Config's exact laws is recorded too, as kcl_exact_residual and
    kcl_exact_residual_rel.

    The history records how the operating point was reached: method (op_method) and fallbacks, every fallback ngspice
    tried with its outcome (fallback_stages), with the first 20 of its notes_count stderr notes. Its iteration count and
    timings come from ngspice's rusage (see _RUSAGE), which counts fallback iterations too.
    """
    name = "spice"
    default_tol = 1e-3
    default_max_iterations = 100

    default_diode = "D"
    SOLVER_OPTIONS = ("matrix_solver", "diode", "ron")  # the options that are not ngspice's

    def __init__(self, tol=None, max_iterations=None, matrix_solver="klu", diode=None, ron=1e-3, **options):
        diode = self.default_diode if diode is None else diode
        if matrix_solver not in ("klu", "sparse"):
            raise ValueError(f"matrix_solver must be 'klu' or 'sparse', not {matrix_solver!r}")
        if diode not in DIODE_MODELS:
            raise ValueError(f"diode must be one of {DIODE_MODELS}, not {diode!r}")
        if not (isinstance(ron, (int, float)) and not isinstance(ron, bool) and ron > 0):
            raise ValueError(f"ron must be a positive resistance, not {ron!r}")
        super().__init__(tol, max_iterations, matrix_solver=matrix_solver, diode=diode, ron=ron, **options)
        ngspice()  # load and initialize the shared library here, outside the timed solve calls

    def describe(self):
        return dict(name=self.name, reltol=self.tol, itl1=self.max_iterations,
                    itl1_effective=max(100, self.max_iterations), **self.options)

    def spice_options(self):
        extra = {k: v for k, v in self.options.items() if k not in self.SOLVER_OPTIONS}
        return [f"reltol={float(self.tol)!r}", f"itl1={int(self.max_iterations)}", self.options["matrix_solver"]] + [
            k if v is True else f"{k}={v}" for k, v in extra.items()]

    def solve(self, cfg):
        ng = ngspice()
        history = {}
        raw = Path(tempfile.gettempdir()) / f"crossbar_op_{os.getpid()}.raw"
        t = time.perf_counter()
        text = netlist(cfg, self.spice_options(), self.options["diode"], self.options["ron"])
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
            history["notes_count"] = len(notes)
            history["fallbacks"] = fallback_stages(notes)
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
            current = as_solved_current(cfg, self.options["diode"], self.options["ron"],
                                        float(self.options.get("gmin", 1e-12)))
            v, i, ok, history["kcl_residual"], history["kcl_residual_rel"] = kcl_check(cfg, u, self.tol, abstol,
                                                                                       current=current)
            if current is not None:
                _, _, _, history["kcl_exact_residual"], history["kcl_exact_residual_rel"] = kcl_check(cfg, u, self.tol,
                                                                                                    abstol)
            history["verify_time"] = time.perf_counter() - t
            return SolveOutput(v, i, ok, iterations, history)
        finally:
            for command in ("destroy all", "remcirc"):
                try:
                    ng.exec_command(command)
                except NgSpiceCommandError:
                    pass
            raw.unlink(missing_ok=True)


class SpiceExpSolver(SpiceSolver):
    """SpiceSolver writing Shockley diodes as behavioral exp() sources, without the diode element's limiting."""
    name = "spice_exp"
    default_diode = "exp"
