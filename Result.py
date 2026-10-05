import math
from dataclasses import dataclass, field

import numpy as np

from Config import Config

CONVERGED = "converged"
NOT_CONVERGED = "not_converged"
ERROR = "error"
# Only in run records: the solver's process died, or ran past the time limit, before returning a Result.
CRASHED = "crashed"
TIMEOUT = "timeout"


@dataclass
class Result:
    """
    This class represents the result of a simulation study. It contains the following information:
    - the Config used for the simulation
    - the algorithm that solved it
    - status: CONVERGED, NOT_CONVERGED (ran out of iterations) or ERROR (the solver raised; see `error`)
    - solved voltages across every branch (None unless the solver returned)
    - solved currents through every branch (None unless the solver returned)
    - simulation runtime in seconds (wall-clock time of the solve call)
    - the number of iterations until convergence (NaN if the simulation did not converge)
    - convergence information (per-iteration history reported by the solver)
    """
    config: Config
    algorithm: str
    status: str
    runtime: float
    iterations: float = math.nan
    v: np.ndarray | None = None
    i: np.ndarray | None = None
    history: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def converged(self):
        return self.status == CONVERGED

    def summary(self):
        """JSON-serializable record of this run without the solution vectors; NaN is written as None."""
        return dict(
            algorithm=self.algorithm,
            p=self.config.p,
            q=self.config.q,
            seed=self.config.seed,
            status=self.status,
            runtime=self.runtime,
            iterations=None if math.isnan(self.iterations) else int(self.iterations),
            error=self.error,
            history=self.history,
            config=self.config.params(),
        )
