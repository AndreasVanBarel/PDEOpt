#############################
# Testing module Gradient   #
#############################

using General # Vec, lm
using MLMC # mlmc
using Problems # problem definitions etc
using Stoch
import Gradient
g = Gradient

import Volume_C_Solver
Solver = Volume_C_Solver

## cache
test_cache = g.Cache(3)
g.cache(test_cache, 1, "one")
g.cache(test_cache, 2, "two")
g.retrieve(test_cache, 1)
g.retrieve(test_cache, 3)
g.cache(test_cache, 3, "three")
g.cache(test_cache, 4, "four")
g.retrieve(test_cache, 1)

## gradient
prob = problems1[1]
h = Hierarchy(RegularGrid2D(prob.m0,prob.m0),prob.L)

# Defining a ComputeStruct
getm(ℓ::Int) = (prob.m0-1)*2^ℓ+1
u = zeros(getm(prob.L),getm(prob.L))
function cost_grad_state(u::Matrix{Float64}, k::Matrix{Float64})
    m = size(u,1)
    Δx = Δy = 1/(m-1)
    nodes = LinRange(0,1,m)
    z = prob.zfun.(nodes,nodes')
    Solver.cost_grad_state(u, z, k, Δx, Δy, prob.α)
end
k_sampler_generator(seed::Int) = k_sampler = Stoch.gen_sampler(seed, prob.distribution, h)
#k_sampler_generator(seed::Int) = (ℓ::Int,i::Int)->exp.(Stoch.gen_sampler(seed, prob.stochfield, h)(ℓ,i)) #NOTE: Extremely slow since Stoch.gen_sampler is called every time!!
restrict_k = General.inject
lm_u(u,ℓ) = General.lm(u, getm(ℓ), getm(ℓ))
lm_y(y,ℓ) = General.lm(y, getm(ℓ), getm(ℓ))
lm_∇J(∇J,ℓ) = General.lm(∇J, getm(ℓ), getm(ℓ))
costs = [2^(2.26ℓ) for ℓ in 0:prob.L]
c = g.ComputeStruct(cost_grad_state, k_sampler_generator, restrict_k, lm_u, lm_y, lm_∇J, prob.L, costs, MLMC.AggregateRMSE())

_,_,_,_,samplingdata = c(u,0.0001)
c(u,samplingdata)
