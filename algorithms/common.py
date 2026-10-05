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
