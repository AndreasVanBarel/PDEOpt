# This file sets up the experiment variables for the Dirichlet to Neumann control problem.

using Problems # problem (hyper)parameters
using General # contains level mapping functions etc
using Gradient
using Stoch
import Dirichlet_To_Neumann_Solver
import MLMC

prob = Problems.problems2[4]
#prob.L = 5;
Solver = Dirichlet_To_Neumann_Solver

# Construct Hierarchy h
h = Hierarchy(RegularGrid2D(prob.m0,prob.m0),prob.L)
h_u = Hierarchy(RegularGrid1D(prob.m0),prob.L)

# Defining a ComputeStruct
getm(ℓ::Int) = (prob.m0-1)*2^ℓ+1
function cost_grad_state(u::Vector{Float64}, k::Matrix{Float64})
    m = size(u,1)
    Δx = Δy = 1/(m-1)
    nodes = LinRange(0,1,m)
    φ = prob.φfun.(nodes)
    Solver.cost_grad_state(u, φ, k, Δx, Δy, prob.α)
end

# Defining the sampler; moving the uncertainty away from the boundary
function k_sampler_generator(seed::Int)
    k_sampler = Stoch.gen_sampler(seed, prob.distribution, h)
    function k_sampler_modified(ℓ,i)
        k_sample = k_sampler(ℓ,i)
        f(x) = x^2/(x+1)
        k_sample[:,1:round(Int,getm(ℓ)/4)].=1
        return k_sample
    end
    return k_sampler_modified
end

#k_sampler_generator(seed::Int) = (ℓ::Int,i::Int)->exp.(Stoch.gen_sampler(seed, prob.stochfield, h)(ℓ,i)) #NOTE: Extremely slow since Stoch.gen_sampler is called every time!!
restrict_k = General.inject
lm_endzeros(v::Vector{Float64},n::Int) = General.lm([0.0;v;0.0],n)[2:end-1]
lm_u(u,ℓ) = lm_endzeros(u, getm(ℓ))
lm_y(y,ℓ) = General.lm(y, getm(ℓ), getm(ℓ))
lm_∇J(∇J,ℓ) = lm_endzeros(∇J, getm(ℓ))
costs = [2^(2.26ℓ) for ℓ in 0:prob.L]
c = ComputeStruct(cost_grad_state, k_sampler_generator, restrict_k, lm_u, lm_y, lm_∇J, prob.L, costs, MLMC.AggregateRMSE())
norm_∇J(g) = norm([0.0;g;0.0],h_u.meshes[end])
inner_∇J(a,b) = inner([0.0;a;0.0],[0.0;b;0.0],h_u.meshes[end])
ls_options = (method="quadmin",)

# For MG/OPT
cs = [ComputeStruct(cost_grad_state, k_sampler_generator, restrict_k, lm_u, lm_y, lm_∇J, ℓ, costs[1:ℓ+1], MLMC.AggregateRMSE()) for ℓ in 0:prob.L]
lm_mgopt = lm_u
ls_options_smoother = (method="quadmin",)
ls_options_smoother_fallback = (method="armijo",inner=inner_∇J,s_guess=8.0,s_decay=0.25,print=0,max_evals=10)
smoother(f,u,μ,k) = Optimization.smooth(f, u, μ; norm_∇J=norm_∇J, print=2, breakcond=g->norm_∇J(g)<1e-6, ls_options=ls_options_smoother, ls_options_fallback=ls_options_smoother_fallback)

#ls_options_mgopt = (method="constant",stepsize=1.0) # cheapest
ls_options_mgopt = (method="armijo",inner=inner_∇J,s_guess=1.0,s_decay=0.5) # This ensures descent happens in the MG/OPT iteration, providing the method with some globalization properties
#ls_options_mgopt = (method="quadmin",); # potentially fastest descent

# number of NCG iterations given MG/OPT level k
μ_pre = 2*[(2^(prob.L-k) for k in 1:prob.L)...,0]; μ_post = 2*[(2^(prob.L-k) for k in 1:prob.L)...,1]

# initial guess
u0 = zeros(getm(prob.L)-2)

 # inspection functions
 function plot_solution(J,∇J,𝔼y,𝕍y,ϵ)
     m_y = h.meshes[end]
     m_u = h_u.meshes[end]
     println("J = $J, ‖∇J‖ = $(norm_∇J(∇J))")
     pp.newfig(1); pp.plot(m_u,[0;u;0],1)
     pp.newfig(2); pp.plot(m_u,[0;∇J;0],2)
     pp.newfig(3); pp.surf(m_y,𝔼y,3)
     pp.newfig(4); pp.surf(m_y,𝕍y,4)
 end

 # generates a dict for exporting
 function export_solution(J,∇J,𝔼y,𝕍y,ϵ)
     m_y = h.meshes[end]
     m_u = h_u.meshes[end]
     data = Dict("u"=>u,
     "J"=>J,"grad"=>∇J,"nodes"=>m_u.nodes_x,"nodes1"=>m_y.nodes_x,"nodes2"=>m_y.nodes_y,
     "Ey"=>𝔼y,"Vy"=>𝕍y,"eps"=>ϵ)
     toMatlab(data, "sol_DN")
 end
