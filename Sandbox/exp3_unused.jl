## EXPERIMENT 3 ##
# Basic MG/OPT fixed sample experiments #
using Gradient
using Optimization

"ASSUME:
 cs       - A Vector{ComputeStruct} (see Gradient.ComputeStruct)
 u0       - A compatible starting value for the iteration, given at the finest
 prob     - A Problem
 norm_∇J  - An inner product for the gradient
 lm_mgopt - A level mapping function for MG/OPT
 smoother - A smoother for MG/OPT
 ls_options_mgopt - Specifications for the linesearch in MG/OPT"

K = prob.L
q = 1.0/16 # amount of samples to retain on a courser level
n = [128,64,32,16,8,4]
# n = 1*[5551, 949, 158, 32, 6, 1][end-K:end]
# n = fill(1,prob.L+1)
ss = [SamplingData(0, ceil.(Int, n[1:ℓ+1]*q^(K-ℓ))) for ℓ in 0:K]
#ss = [SamplingData(ℓ, ceil.(Int, n[1:ℓ+1]*q^(K-ℓ))) for ℓ in 0:K]
f = gen_f(cs, ss)
s = ss[end]

## 1.1 Checks the basic functioning of the MG/OPT V-cycle.
global u, gnorm, η
u = u0
@time u,gnorm,η = mgoptV(u,zero(u),f,lm_mgopt,K,μ_pre,μ_post; smooth=smoother, print=2, inner=inner_∇J, ls_options=ls_options_mgopt)

## 1.2 Checks the convergence speed of the V-cycle
u = u0
us = [u]; gnorms = [norm(f(u,K)[2])]; ηs = [NaN]
for i=1:5
    global u
    println("==== V-cycle $i ====")
    @time u,gnorm,η = mgoptV(u,zero(u),f,lm_mgopt,K,μ_pre,μ_post; smooth=smoother, print=2, inner=inner_∇J, ls_options=ls_options_mgopt)
    println("gnorm = $gnorm")
    push!(us,u); push!(gnorms,gnorm); push!(ηs,η)
end
pp.newfig(10)
pp.semilogy(1:length(gnorms),gnorms)

## 2.1 iterated restarted NCG convergence...
# u = Fun(x->0.0, h.meshes[end])
# compute = ComputeStruct(problem,h)
# us = [u]; gnorms = [norm(compute(u,ss[end])[2])]
# for i=1:5
#     global u
#     print("NCG $i: ")
#     u = ncg(u, compute, 1e-10, 2, ss[end]; plots=false)
#     gnorm = norm(compute(u,ss[end])[2])
#     println("gnorm = $gnorm")
#     push!(us,u); push!(gnorms,gnorm)
# end

## 3.1 Checks the basic functioning of the MG/OPT W-cycle.
u = u0
@time u,gnorm,η = mgoptW(u,zero(u),f,lm_mgopt,K,[fill(2,K);0],fill(2,K+1); smooth=smoother, print=2, inner=inner_∇J, ls_options=ls_options_mgopt)

## 3.2 Checks the convergence speed of the W-cycle
u = u0
us = [u]; gnorms = [norm(f(u,K)[2])]; ηs = [NaN]
for i=1:5
    global u
    println("==== W-cycle $i ====")
    u,gnorm,η = mgoptW(u,zero(u),f,lm_mgopt,K,[fill(2,K);0],fill(2,K+1); smooth=smoother, print=2, inner=inner_∇J, ls_options=ls_options_mgopt)
    println("gnorm = $gnorm")
    push!(us,u); push!(gnorms,gnorm); push!(ηs,η)
end
pp.newfig(10)
pp.semilogy(1:length(gnorms),gnorms)

## 4.1 Finding problematic iterations
u = u0
try
    global u, gnorm, η
    u,gnorm,η = mgoptV(u,zero(u),f,lm_mgopt,K,μ_pre,μ_post; smooth=smoother, print=2, inner=inner_∇J, ls_options=ls_options_mgopt)
catch e
    global ex = e
end

v_p = ex.d.v
u_p = bottom(ex).d.u
∇J_p = bottom(ex).d.∇J
