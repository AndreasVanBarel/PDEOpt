## EXPERIMENT 1 ##
# Basic NCG fixed sample experiment #
# This gives an indication of the convergence speed of the optimization algorithm itself.

using Gradient
using Optimization

"ASSUME:
 c       - A ComputeStruct (see Gradient.~)
 u0      - A compatible starting value for the iteration."

# Experiment parameters
maxit = 25
n = 2 .^(prob.L+2:-1:2)

# Experiment
s = SamplingData(0,n)
its = []
@time u = ncg(u0, c, 1e-10, maxit, s; norm=norm_∇J, print=2, save=its, ls_options=ls_options, Π=prob.Π)

# Convergence plot
gradnorms = [norm_∇J(its[i][4]) for i in 1:length(its)]
ϵs = [its[i][9] for i in 1:length(its)]
pp.newfig(1);
pp.semilogy(1:length(its), gradnorms)
pp.semilogy(1:length(its), ϵs)

# Reevaluation using new samples
s2 = SamplingData(1,n)
∇Js2 = [c(its[i][2],s2)[2] for i in 1:length(its)]
pp.semilogy(1:length(its),[norm_∇J(∇J) for ∇J in ∇Js2])
