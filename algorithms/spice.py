from algorithms.common import Solver


class SpiceSolver(Solver):
    """DC operating point of the crossbar netlist via ngspice (PySpice)."""
    name = "spice"

    def solve(self, cfg):
        raise NotImplementedError("SPICE solver not implemented yet")
