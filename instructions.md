Let's implement the solve method in SpingarnSolver.

Let:
- $v$ denote a vector of branch voltages
- $i$ denote a vector of branch currents
- $\Gamma$ is a diagonal matrix of preconditioning weights, to be defined.
- $A$ is the incidence matrix, such that $v \in range(A^T)$, $i \in ker(A)$
- $M_\Gamma = A^T(A \Gamma^{-1} A^T)^{-1}A\Gamma^{-1}$ is the projector onto $range(A^T)$ with the $\Gamma$ weighted norm
- $S_\Gamma = 2M_\Gamma - I$ is the reflection operator in the $\Gamma$ space
- $\alpha$ is a tunable parameter in $(0, 1]$
- $T_e$ for all branches e is the branch relation operator. Specifically, $i_e \in T_e(v_e)$ if and only if $(v_e, i_e) \in B_e$, where $B_e$ is the behaviour of the branch. For a linear resistor, this is $B_e = \{(v, i) | v = i*r\}$. For our nonlinear resistors, $B_e = \{(v, i) | i = v/R_{max} + (1/R_{min} - 1/R_{max})\tanh(v)\}$. We may want to incorporate other elements in the future such as diodes.

given initial condition $a_0=0$, the iteration is:

while not converged, $k = 0, 1, 2, \dots$ :
1) for each branch $e$: $v_e^k = J_{\gamma_eT_e}(a_e^k)$ and $b_e^k = 2v_e^k - a_e^k$
2) $a^{k+1} = (1-\alpha)a^k + \alpha S_\Gamma b^k$

output: $v^k$ and $i_k = \frac{1}{2} \Gamma^{-1}(a^k - b^k)$

$J_T$ is the resolvent of $T$ i.e. $J_T = (I + T)^{-1}$
