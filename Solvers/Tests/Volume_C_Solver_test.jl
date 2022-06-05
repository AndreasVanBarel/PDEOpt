##################################
# Testing module Volume_C_Solver #
##################################

import Volume_C_Solver
Solver = Volume_C_Solver

n = 33
nodes = LinRange(0,1,n)
#mesh = Mesh(nodes,nodes)

# variables
y = zeros(n,n)
f = y[:,:]
k = ones(n,n)
u = sin.(LinRange(0,π,n))*sin.(LinRange(0,π,n))'
Δx = Δy = 1/(n-1)
z = sin.(LinRange(0,π,n))*sin.(LinRange(0,π,n))'
#z = zeros(n,n)

Solver.solve_state!(y,u,k,Δx,Δy)
p = Solver.solve_adjoint(y,u,z,k,Δx,Δy)

# costfun and gradient
α = 0
J = Solver.cost(u,z,k,Δx,Δy,α)
J,∇J,y = Solver.cost_grad_state(u,z,k,Δx,Δy,α)

# test gradient by comparing to finite differences result.
ϵ = 1e-8 #perturbation used in finite differences
∇J_diff = Array{Float64}(undef,size(u))
J = u->Solver.cost(u,z,k,Δx,Δy,α)
for i = 1:length(u) #linear indexing
    δu = zeros(size(u)); δu[i] = ϵ
    ∇J_diff[i] = (J(u + δu) - J(u))/ϵ
end
∇J_diff .= ∇J_diff.*length(u) #ensure correct scaling
e = ∇J .- ∇J_diff

## DEVELOPMENT This is part of finding an error in the gradient data for some given v,u,∇J obtained from a no descent failure during a V-cycle.
n = 33
nodes = LinRange(0,1,n)
grid = RegularGrid2D(n,n)
Δx = Δy = 1/(n-1)
z = prob.zfun.(nodes,nodes')
sampler = Stoch.gen_sampler(0, prob.distribution, grid)
k = sampler(1)

d = load("Data\\Volume_C_Solver_testingdata1.jld")
u = d["u"]
v = d["v"]

# Gradient calculated using the adjoint method
J,∇J,y = Solver.cost_grad_state(u,z,k,Δx,Δy,prob.α)

# Gradient calculated using the
ϵ = 1e-8 #perturbation used in finite differences
∇J_diff = Array{Float64}(undef,size(u))
J = u->Solver.cost(u,z,k,Δx,Δy,prob.α)
for i = 1:length(u) #linear indexing
    δu = zeros(size(u)); δu[i] = ϵ
    ∇J_diff[i] = (J(u + δu) - J(u))/ϵ
end
∇J_diff .= ∇J_diff.*(n-1)^2 #ensure correct scaling
e = ∇J .- ∇J_diff
pp.newfig(1); pp.surf(grid,e)
pp.newfig(2); pp.surf(grid,∇J_diff)
pp.newfig(3); pp.surf(grid,v)
pp.newfig(4); pp.surf(grid,∇J.-v)

## Discretization error investigation
import Volume_C_Solver
Solver = Volume_C_Solver
using General
using Plotter

m0 = 5
L = 6
L_ref = L+2 #2^2 × 2^2 times finer

# Optimization problem definition
zfun = (x...)->Float64(all(0.25.<=x.<=0.75)) # target function
#zfun = (x,y)->0.25*(1-cos(2π*x))*(1-cos(2π*y))
α = 1e-5 # cost function parameter
getm(ℓ::Int) = (m0-1)*2^ℓ+1
m_ref = getm(L_ref)
u_ref = 5 .*(1 .-cos.(2π.*range(0,1,length=m_ref))).*(1 .-cos.(2π.*range(0,1,length=m_ref)))'
k_ref = ones(m_ref,m_ref)
z_ref = zfun.(LinRange(0,1,m_ref),LinRange(0,1,m_ref)')
us = [lm(u_ref,getm(ℓ),getm(ℓ)) for ℓ in 0:L]
ks = [inject(k_ref,L_ref,ℓ) for ℓ in 0:L]
zs = [lm(z_ref,getm(ℓ),getm(ℓ)) for ℓ in 0:L]

# Construct hierarchy mesh (don't edit)
h = Hierarchy(RegularGrid2D(m0,m0),L)
mesh_ref = RegularGrid2D(m_ref,m_ref)

# Function generating the cost, gradient and state
function cost_grad_state(u::Matrix{Float64}, k::Matrix{Float64}, z::Matrix{Float64})
    m = size(u,1)
    Δx = Δy = 1/(m-1)
    nodes = LinRange(0,1,m)
    z = zfun.(nodes,nodes')
    Solver.cost_grad_state(u, z, k, Δx, Δy, α)
end

J_ref,∇J_ref,y_ref = cost_grad_state(u_ref, k_ref)

y_diffs = Vector{typeof(y_ref)}(undef,L+1)
∇J_diffs = Vector{typeof(∇J_ref)}(undef,L+1)

for ℓ=0:L
    J, ∇J, y = cost_grad_state(us[ℓ+1], ks[ℓ+1], zs[ℓ+1])
    y_diffs[ℓ+1] = y-inject(y_ref,L_ref,ℓ)
    ∇J_diffs[ℓ+1] = ∇J-inject(∇J_ref,L_ref,ℓ)
end

y_bias = sqrt.([sum(y_diffs[ℓ+1].^2)/length(ydiffs[ℓ+1]) for ℓ = 0:L])
∇J_bias = sqrt.([sum(∇J_diffs[ℓ+1].^2)/length(∇Jdiffs[ℓ+1]) for ℓ = 0:L])

pp.newfig(1)
pp.plot(mesh_ref,∇J_diffs[end],1); pp.colorbar()
