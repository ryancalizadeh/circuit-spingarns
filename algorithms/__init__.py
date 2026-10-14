from algorithms.condat_vu import CondatVuSolver
from algorithms.scellier import ScellierLinesSolver, ScellierSolver
from algorithms.spice import SpiceSolver
from algorithms.spingarns import SpingarnCholmodSolver, SpingarnSectorSolver, SpingarnSLUSolver, SpingarnSolver

# Every solver the experiment runner can select by name. The order fixes each algorithm's plot color.
SOLVERS = {cls.name: cls for cls in [SpingarnSolver, SpiceSolver, SpingarnSectorSolver, SpingarnSLUSolver,
                                     SpingarnCholmodSolver, CondatVuSolver, ScellierSolver, ScellierLinesSolver]}
