## Some quick and simplified experiments for the Dirichlet_To_Neumann problem.

# Performing NCG optimization on Dirichlet to Neumann problem
using Optimization
using General
import Dirichlet_To_Neumann_Solver
Solver = Dirichlet_To_Neumann_Solver

n = 513 #number of discretization points in one dimension (including boundaries)

kfun = (x1,x2)->1.0
#φfun = x->sin(π*x)
φfun = x->(0.25<=x)&&(x<=0.75) ? 1.0 : 0.0 #blockfun
ufun = x->0*sin(π*x)
α = 0e-3

nodes = LinRange(0,1,n)
nodes_internal = view(nodes,2:n-1)
φ = φfun.(nodes_internal)
k = kfun.(nodes,nodes')
u = ufun.(nodes_internal)
Δx = Δy = 1/(n-1)

## EXP1: Deterministic single level NCG experiment ##
f = u->Solver.cost_grad_state(u,φ,k,Δx,Δy,α)[1:2]
u_opt = ncg(u,f,1e-8,25,print=2)
pp.newfig(2)
pp.plot(nodes[2:end-1],u_opt)

## EXP2: MG/OPT experiments ##
K = 3 #4 levels
n0 = 65
ns = [(n0-1)*2^ℓ+1 for ℓ in 0:K]
n = ns[end]

nodes = LinRange(0,1,n)
nodes_int = view(nodes,2:n-1)
u = ufun.(nodes_int)

lm1(u::Vector{T},k::Int) where T = lm([0;u;0],ns[k+1])[2:end-1] # level mapping function for MG/OPT

function f1(u,k::Int) # costfun gradient function for MG/OPT
    n = ns[k+1]
    length(u)==n-2 || @error("u is represented on the wrong level!")
    nodes = LinRange(0,1,n)
    nodes_int = view(nodes,2:n-1)
    φ = φfun.(nodes_int)
    k = kfun.(nodes,nodes')
    Δx = Δy = 1/(n-1)
    Solver.cost_grad_state(u,φ,k,Δx,Δy,α)[1:2]
end

u,ngrad,η = mgoptV(u,zeros(ns[end]-2),f1,lm1,K;norm=x->norm(x)/sqrt(length(x)), print=true)

pp.newfig(4)
pp.plot(nodes[2:end-1],u)
