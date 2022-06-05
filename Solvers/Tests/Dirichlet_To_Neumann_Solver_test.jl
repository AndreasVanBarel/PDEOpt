##############################################
# Testing module Dirichlet_To_Neumann_Solver #
##############################################

import Dirichlet_To_Neumann_Solver
Solver = Dirichlet_To_Neumann_Solver
#using General

n = 33
nodes = LinRange(0,1,n)
#mesh = Mesh(nodes,nodes)

# variables
y = zeros(n,n)
f = y[:,:]
k = ones(n,n)
#u = ones(n-2)
u = sin.(LinRange(0,π,n))[2:end-1]
Δx = Δy = 1/(n-1)
φ = sin.(LinRange(0,π,n))[2:end-1]
#φ = zeros(n-2)

Solver.solve_state!(y,u,f,k,Δx,Δy)
p = Solver.solve_adjoint(y,u,φ,k,Δx,Δy)

# costfun and gradient
α = 0
J = Solver.cost(u,φ,k,Δx,Δy,α)
J,∇J,y = Solver.cost_grad_state(u,φ,k,Δx,Δy,α)

# test gradient by comparing to finite differences result.
ϵ = 1e-8 #perturbation used in finite differences
∇J_diff = Array{Float64}(undef,size(u))
J = u->Solver.cost(u,φ,k,Δx,Δy,α)
for i = 1:length(u)
    δu = zeros(size(u)); δu[i] = ϵ
    ∇J_diff[i] = (J(u + δu) - J(u))/ϵ
end
∇J_diff .= ∇J_diff.*(n-1)
e = ∇J .- ∇J_diff
