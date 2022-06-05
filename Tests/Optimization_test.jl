using Optimization
using General
using Gradient
using Problems
using MLMC
using Stoch

import Volume_C_Solver
Solver = Volume_C_Solver

## This test checks the provided costfunction/gradient pair for consistency.

#f - function yielding (J,∇J) pair
#u - input value
#d - direction in which to compute. If empty, chooses gradient in u
function compute_line(f::Function, u, d)
    g = u->f(u)[2]
    J,∇J = f(u)
    s = quadmin(u,∇J,d,g)[1]
    steps = range(0, length = 20, stop = 2s)
    Js∇Js = [f(u+d*step) for step in steps]
    Js = [t[1] for t in Js∇Js]
    ∇Js = [t[2] for t in Js∇Js]
    return steps, Js, ∇Js
end
compute_line(f::Function, u) = compute_line(f,u,-f(u)[2])

L = 0
prob = problems1[1]
h = Hierarchy(RegularGrid2D(prob.m0,prob.m0),prob.L)

# Defining a ComputeStruct
getm(ℓ::Int) = (prob.m0-1)*2^ℓ+1
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
compute = ComputeStruct(cost_grad_state, k_sampler_generator, restrict_k, lm_u, lm_y, lm_∇J, prob.L, costs, MLMC.AggregateRMSE())

f = u->compute(u,SamplingData(0,[10]))[1:2]
u = zeros(getm(prob.L),getm(prob.L)) # Starting value for u given on the finest level
d = -f(u)[2]
steps,Js,∇Js = compute_line(f,u,d)
∇Jds = dot.(∇Js, [d/norm(d)])

pp.newfig(1)
pp.plot(steps,[Js ∇Jds])
pp.newfig(2)
pp.plot(steps,∇Jds)

## MG/OPT TESTS ##
using Solver
using General
using Gradient
using Optimization
using Printf

K = 1 # two level MG/OPT tests
problem = problems[7]
h = Hierarchy(problem, K)
LM = gen_LM(h) #level mapping function

n = [800,160]
q = 1.0/16
ss = [SamplingData(0, ceil.(Int, n[1:ℓ+1]*q^(K-ℓ))) for ℓ in 0:K]
f = gen_f(ss, problem, h)
u = Fun(x->0.0, h.meshes[end])

# generation of a reference solution using NCG
compute = ComputeStruct(problem,h)
its = []
uref = ncg(u, compute, 1e-10, 50, ss[end]; plots=false, save=its)

# MG/OPT
u,gnorm,η = mgoptVdebug(u,zero(u),f,LM,K,K; print=true)

for i = 1:5
    global u
    u,gnorm,η = mgoptVdebug(u,zero(u),f,LM,K,K; print=true)
end
