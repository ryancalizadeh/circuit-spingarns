class Result:
    """
    This class represents the result of a simulation study. It contains the following information:
    - the Config used for the simulation
    - solved voltages across every branch
    - solved currents through every branch
    - simulation runtime
    - (optional) the number of iterations until convergence (NaN if the simulation did not converge)
    - (optional) convergence information
    """