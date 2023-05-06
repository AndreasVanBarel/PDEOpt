# PDEOpt

This Julia code performs robust optimization of PDE constrained systems for a number of test problems. By constructing a multilevel Monte Carlo (MLMC) hierarchy for the Gradient, the optimization process can be sped up, as discussed in [A. Van Barel, S. Vandewalle, 2019](https://arxiv.org/pdf/1711.02574). One can additionally fit these MLMC gradients in an MG/OPT hierarchy to further improve the convergence speed of the optimization procedure, see [A. Van Barel, S. Vandewalle, 2021](https://arxiv.org/pdf/2006.01231).

## Running experiments
First, ensure that `pwd()` returns the path to the PDEOpt local repository.
This could be done by executing a line such as `cd("C:\\Users\\SomeUser\\PDEOpt")`.
Then, run `init.jl`, which activates the right package environment, loads some dependencies, and adds relevant files to the path.

The experiments are set up by running one of the following scripts:

 - `Experiments/exp_V_setup.jl` for the ubiquitous elliptic model problem
 - `Experiments/exp_DN_setup.jl` for the Dirichlet to Neumann model problem
 - `Experiments/exp_B_setup.jl` for the Burgers' equation model problem

Note that at the top of these files, specific problem parameters are chosen by the line `prob = Problems.problems1[1]`. A different test case may be chosen by editing this line to, e.g., `prob = Problems.problems1[2]`. The `Problem` module stores the various test cases and can be extended with new ones.

A specific experiment is then carried out by running one of the following scripts:

 - `Experiments/exp1.jl`: Basic NCG fixed sample experiment, giving an indication of the convergence speed of the optimization algorithm itself.
 - `Experiments/exp2.jl`: MLMC with finest level NCG as described in [A. Van Barel, S. Vandewalle, (2019)](https://arxiv.org/pdf/1711.02574).
 - `Experiments/exp3.jl`: MG/OPT + MLMC as described in [A. Van Barel, S. Vandewalle, (2021)](https://arxiv.org/pdf/2006.01231).
 - `Experiments/exp3_W.jl` - Same as above but with W-cycles instead of V-cycles. Results were published in [my PhD (2021)](https://lirias.kuleuven.be/retrieve/638063).

## Other PDE constraints
For completely different PDE constraints, a new solver must be provided following the ones present in `Solvers\`. Broadly speaking, a solver must contain a routine for solving the forward PDE, the corresponding adjoint PDE, and for evaluating the cost function. The existing mapping operators (prolongations and restrictions) can be reused, or new ones may be defined and set during setup.
