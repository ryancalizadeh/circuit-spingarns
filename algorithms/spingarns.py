from algorithms.common import Solver


class SpingarnSolver(Solver):
    """Spingarn's method of partial inverses with impedance-matching preconditioning."""
    name = "spingarn"

    def solve(self, cfg):
        raise NotImplementedError("Spingarn solver not implemented yet")
