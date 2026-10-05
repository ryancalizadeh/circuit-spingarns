from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolveOutput:
    """
    What a solver hands back to the experiment runner. The runner does the timing and wraps this into a Result.

    v, i: branch voltages and currents for all m branches, in Config's branch order and sign convention.
    converged: whether the solver met its stopping criterion within its iteration budget.
    iterations: iterations performed (outer iterations for Spingarn, Newton iterations for SPICE).
    history: optional per-iteration convergence information, e.g. {"residual": [...]}; values must be
        JSON-serializable lists or scalars so they can be saved alongside the run summary.
    """
    v: np.ndarray
    i: np.ndarray
    converged: bool
    iterations: int
    history: dict = field(default_factory=dict)


class Solver(ABC):
    """
    Base class for a circuit solver. Subclasses set `name` and implement `solve`. Options shared by every solver
    (tolerance, iteration budget) are fixed at construction so that all algorithms in a sweep get the same ones;
    algorithm-specific options go in **options.
    """
    name = None

    def __init__(self, tol=1e-9, max_iterations=10_000, **options):
        self.tol = tol
        self.max_iterations = max_iterations
        self.options = options

    @abstractmethod
    def solve(self, cfg):
        """Solve the circuit described by `cfg` (a Config) and return a SolveOutput. Called inside the timer."""

    def describe(self):
        """JSON-serializable description of this solver's settings, saved with the experiment."""
        return dict(name=self.name, tol=self.tol, max_iterations=self.max_iterations, **self.options)
