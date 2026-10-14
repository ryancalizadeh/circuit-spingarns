from algorithms.condat_vu import CondatVuSolver
from algorithms.newton import NewtonSafeguardedSolver, NewtonSolver
from algorithms.scellier import ScellierLinesSolver, ScellierSolver
from algorithms.spice import SpiceExpSolver, SpiceSolver
from algorithms.spingarns import (SpingarnAdaptiveSolver, SpingarnAdaptiveSplitSolver, SpingarnCholmodSolver,
                                  SpingarnSectorSolver, SpingarnSLUSolver, SpingarnSolver)

# Every solver the experiment runner can select by name. The order fixes each algorithm's plot color.
SOLVERS = {cls.name: cls for cls in [SpingarnSolver, SpiceSolver, SpingarnSectorSolver, SpingarnSLUSolver,
                                     SpingarnCholmodSolver, CondatVuSolver, ScellierSolver, ScellierLinesSolver,
                                     SpingarnAdaptiveSolver, SpingarnAdaptiveSplitSolver, NewtonSolver,
                                     NewtonSafeguardedSolver, SpiceExpSolver]}
