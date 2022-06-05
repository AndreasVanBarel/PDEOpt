# This file sets up the experiment variables for the Burgers' end state control problem.

using Problems # problem (hyper)parameters
using General # contains level mapping functions etc
using Gradient
using Stoch
import Burgers_Solver
import MLMC

prob = problems3[16]
Solver = Burgers_Solver

# setup variables
kscale = 1e-3
mx,mt = prob.m0
T = prob.T

max_value_in_u = 5.0
#max_value_in_u = (Δx^2 - 2k*Δt)/(Δt*Δx)

# Construct Hierarchy h
h = Hierarchy(RegularGrid(mx,mt))
extend!(h,prob.L,prob.a)

# Defining a ComputeStruct
getmx(ℓ::Int) = (mx-1)*2^ℓ+1
function cost_grad_state(u::Vector{Float64}, k::Vector{Float64})
    m = size(u,1)
    Δx = 1/(m-1)
    Δt = T/(mt-1)
    nodes = LinRange(0,1,m)
    z = prob.zfun.(nodes)
    Solver.cost_grad_state(u, z, k, ones(m).*prob.s, mt, Δx, Δt, prob.α)
end
h_u = Hierarchy(RegularGrid1D(mx),prob.L);
function k_sampler_generator(seed::Int)
    k_sampler = Stoch.gen_sampler(seed, prob.distribution, h_u)
    return (ℓ::Int,i::Int)->kscale.*k_sampler(ℓ,i)
end
restrict_k = General.inject
lm_u(u,ℓ) = General.lm(u, getmx(ℓ))
lm_y(y,ℓ) = General.lm(y, getmx(ℓ), mt)
lm_∇J(∇J,ℓ) = General.lm(∇J, getmx(ℓ))
costs = [2^ℓ for ℓ in 0:prob.L]
c = ComputeStruct(cost_grad_state, k_sampler_generator, restrict_k, lm_u, lm_y, lm_∇J, prob.L, costs, MLMC.AggregateRMSE())
norm_∇J(g) = norm(g,h_u.meshes[end])
inner_∇J(a,b) = inner(a,b,h_u.meshes[end])
#s_bound(u,d) = (max_value_in_u - maximum(abs.(u)))/maximum(abs.(d))
function s_bound(u,d)
    all(u.<=max_value_in_u) || @warn("maximum value in u surpassed!")
    f(u,d,a) = d==0 ? Inf : (d>0 ? (a-u)/d : (-a-u)/d)
    return minimum(f.(u,d,max_value_in_u))
end
ls_options = (method="quadmin",s_min=0.0,s_bound=s_bound)
#ls_options = (method="quadmin",s_min=0.0,s_bound=s_bound)

# For MG/OPT
cs = [ComputeStruct(cost_grad_state, k_sampler_generator, restrict_k, lm_u, lm_y, lm_∇J, ℓ, costs[1:ℓ+1], MLMC.AggregateRMSE()) for ℓ in 0:prob.L]
lm_mgopt = lm_u
#ls_options_smoother = (method="armijo",inner=inner_∇J,s_guess=8.0,s_decay=0.25,print=1,max_evals=10)
ls_options_smoother = (method="quadmin",s_min=0.0,s_max=10.0,s_bound=s_bound)
ls_options_smoother_fallback = (method="armijo",inner=inner_∇J,s_guess=8.0,s_decay=0.25,print=0,max_evals=10,s_bound=s_bound)
smoother(f,u,μ,k) = Optimization.smooth(f, u, μ; norm_∇J=norm_∇J, print=2, breakcond=g->norm_∇J(g)<1e-10, ls_options=ls_options_smoother, ls_options_fallback=ls_options_smoother_fallback)

#ls_options_mgopt = (method="constant",stepsize=1.0) # cheapest
ls_options_mgopt = (method="armijo",inner=inner_∇J,s_guess=1.0,s_decay=0.5) # This ensures descent happens in the MG/OPT iteration, providing the method with some globalization properties
#ls_options_mgopt = (method="quadmin",); # potentially fastest descent

# number of NCG iterations given MG/OPT level k
μ_pre = 2*[(2^(prob.L-k) for k in 1:prob.L)...,0]; μ_post = 2*[(2^(prob.L-k) for k in 1:prob.L)...,1]
#μ_pre = [fill(2,prob.L)...,0]; μ_post = fill(2,prob.L+1)

# initial guess
u0 = zeros(getmx(prob.L))

# inspection functions
s1 = SamplingData(0,fill(1,prob.L+1))
function plot_solution(J,∇J,𝔼y,𝕍y,last)
    m_y = h.meshes[end]
    m_u = h_u.meshes[end]
    println("J = $J, ‖∇J‖ = $(norm_∇J(∇J))")
    pp.newfig(1); pp.plot(m_u,u,1)
    pp.newfig(2); pp.plot(m_u,∇J,2)
    pp.newfig(3); pp.surf(m_y,𝔼y,3); pp.newfig(6); pp.plot(m_y,𝔼y,6); pp.colorbar()
    pp.newfig(4); pp.surf(m_y,𝕍y,4)
    pp.newfig(5); pp.plot(m_u,𝔼y[:,end],5); pp.plot(m_u,[prob.zfun(p.x) for p in m_u])
end

# generates a dict for exporting
function export_solution(J,∇J,𝔼y,𝕍y,ϵ)
    m_y = h.meshes[end]
    m_u = h_u.meshes[end]
    data = Dict("u"=>u,
    "J"=>J,"grad"=>∇J,"nodes"=>m_u.nodes_x,"nodes1"=>m_y.nodes_x,"nodes2"=>m_y.nodes_y,
    "Ey"=>𝔼y,"Vy"=>𝕍y,"eps"=>ϵ)
    toMatlab(data, "sol_B")
end


# Experimental code
# function inject_modified(v::Vector, ℓ::Int, ℓ_new::Int)
#     ℓ_new <= ℓ || @error("Return level must be smaller than the original level.")
#     q = 2^(ℓ-ℓ_new)
#     v_new = v[1:q:end]
#     rate = sqrt(2)
#     v_new[1] = rate^(ℓ-ℓ_new)*v[1]; v_new[end] = rate^(ℓ-ℓ_new)*v[end];
#     boundary_multiplier = sqrt(2)^(prob.L-ℓ_new)
#     v_new[2] = rate^(prob.L-ℓ_new)*v[1+q]; v_new[end-1] = rate^(prob.L-ℓ_new)*v[end-q];
#     return v_new
# end
# restrict_k = inject_modified
