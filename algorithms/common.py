from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolveOutput:
    """
    What a solver hands back to the experiment runner. The runner does the timing and wraps this into a Result.

    v, i: branch voltages and currents for all m branches, in Config's branch order and sign convention; None if the
        solver produced no solution at all.
    converged: whether the solver met its stopping criterion within its iteration budget.
    iterations: iterations performed (outer iterations for Spingarn, Newton iterations for SPICE).
    history: optional per-iteration convergence information, e.g. {"residual": [...]}; values must be
        JSON-serializable lists or scalars so they can be saved alongside the run summary.
    """
    v: np.ndarray | None
    i: np.ndarray | None
    converged: bool
    iterations: int
    history: dict = field(default_factory=dict)


def kcl_check(cfg, u, reltol, abstol, A=None, x0=None, current=None):
    """
    Branch voltages and currents implied by node potentials `u`, and whether they satisfy KCL at every node to SPICE's
    own current tolerance: |sum of currents| <= reltol * (sum of |currents|) + abstol. Currents come from the exact
    branch laws, so the residual measures how far `u` is from a true DC solution. Every solver's answer is judged by
    this test. A is cfg's incidence matrix, if already built; x0 optionally warm-starts the cells' series solve (see
    Config.current), which makes the test cheaper without changing it. current, if given, replaces cfg.current as the
    branch laws (called as current(v, x0)): SPICE's diode element, for one, is not quite the Shockley law.
    """
    A = cfg.incidence() if A is None else A
    v = A.T @ u
    i = (cfg.current if current is None else current)(v, x0)
    residual = np.abs(A @ i)
    scale = abs(A) @ np.abs(i)
    # finite first: overflowed currents would pass inf <= reltol * inf (behavioral exp() diodes on 2026-10-14: ngspice
    # declared convergence at node potentials of 4e71 V, where its exp() clamp makes the diode current flat)
    ok = bool(np.all(np.isfinite(residual)) and np.all(residual <= reltol * scale + abstol))
    return v, i, ok, float(residual.max()), float(np.max(residual / np.maximum(scale, abstol)))


class Solver(ABC):
    """
    Base class for a circuit solver. Subclasses set `name` and implement `solve`.

    Every solver takes a tolerance and an iteration budget, but their meaning is solver-specific (e.g. SPICE's RELTOL
    and ITL1), so each subclass sets its own defaults; passing None selects them. Algorithm-specific options go in
    **options.
    """
    name = None
    default_tol = 1e-9
    default_max_iterations = 10_000

    def __init__(self, tol=None, max_iterations=None, **options):
        self.tol = self.default_tol if tol is None else tol
        self.max_iterations = self.default_max_iterations if max_iterations is None else max_iterations
        self.options = options

    @abstractmethod
    def solve(self, cfg):
        """Solve the circuit described by `cfg` (a Config) and return a SolveOutput. Called inside the timer."""

    def describe(self):
        """JSON-serializable description of this solver's settings, saved with the experiment."""
        return dict(name=self.name, tol=self.tol, max_iterations=self.max_iterations, **self.options)
