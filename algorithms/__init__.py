from algorithms.spice import SpiceSolver
from algorithms.spingarns import SpingarnSolver

# Every solver the experiment runner can select by name. The order fixes each algorithm's plot color.
SOLVERS = {cls.name: cls for cls in [SpingarnSolver, SpiceSolver]}
