# Contains code to represent optimization problems
module Problems

export Problem, problems1, problems2, problems3

import Base: show, propertynames, getproperty
using Stoch # Contains types to describe the problem's stochasticity

# solvers
import Dirichlet_To_Neumann_Solver
import Volume_C_Solver

# General
zerofun(x...) = 0.0
id(x)=x #identity
blockfun(x...) = Float64(all(0.25.<=x.<=0.75)) #blockfunction in any dimension

# Problem data
abstract type Problem end

# Problem data accompanying Volume_C PDE constraint
mutable struct Problem1 <: Problem
    α::Float64 # Cost function parameter
    distribution::Distribution # Distribution of the stochastic field
    seed::Int # for all random number generation
    m0::Int # number of nodes in each dimension on coarsest levels
    L::Int # maximum level. The number of levels is L+1.
    zfun::Function #defines target function; (Real,Real)->Real
end
show(io::IO, p::Problem1) = print(io, "Problem 1 (Volume Control) with parameters:
 - α: $(p.α)
 - distribution: $(p.distribution)
 - seed: $(p.seed)
 - m0: $(p.m0)
 - L: $(p.L)
 - zfun: $(p.zfun)")
 propertynames(prob::Problem1) = (fieldnames(Type{Problem1})..., :Π)
 getproperty(prob::Problem1, s::Symbol) = s == :Π ? id : getfield(prob,s)

 # Problem data accompanying Dirichlet_To_Neumann PDE constraint
 mutable struct Problem2 <: Problem
     α::Float64 # Cost function parameter
     distribution::Distribution # Distribution of the stochastic field
     seed::Int #for all random number generation
     m0::Int # number of nodes in each dimension on coarsest levels
     L::Int # maximum level. The number of levels is L+1.
     φfun::Function #defines target function for the flux; Real->Real
 end
 show(io::IO, p::Problem2) = print(io, "Problem 2 (Dirichlet to Neumann Control) with parameters:
  - α: $(p.α)
  - distribution: $(p.distribution)
  - seed: $(p.seed)
  - m0: $(p.m0)
  - L: $(p.L)
  - φfun: $(p.φfun)")
  propertynames(prob::Problem2) = (fieldnames(Type{Problem2})..., :Π)
  getproperty(prob::Problem2, s::Symbol) = s == :Π ? id : getfield(prob,s)

  # Problem data accompanying Burgers' PDE constraint
  mutable struct Problem3 <: Problem
      α::Float64 # Cost function parameter
      distribution::Distribution # Distribution of the stochastic field
      s::Float64 #advection term
      T::Float64 #final time
      seed::Int #for all random number generation
      m0::Tuple{Int,Int} # number of nodes in the spatial and time dimension on coarsest level.
      a::Tuple{Int,Int} # coarsening factor for spatial and temporal dimension.
      L::Int # maximum level. The number of levels is L+1.
      zfun::Function #defines target end state function; Real->Real
      Π::Function #defines a projection of u onto the admissible set
  end
  Problem3(α,distribution,s,T,seed,m0,a,L,zfun) = Problem3(α,distribution,s,T,seed,m0,a,L,zfun,id)
  show(io::IO, p::Problem3) = print(io, "Problem 3 (Burgers' end state control) with parameters:
   - α: $(p.α)
   - distribution: $(p.distribution)
   - s: $(p.s)
   - T: $(p.T)
   - seed: $(p.seed)
   - m0: $(p.m0)
   - a: $(p.a)
   - L: $(p.L)
   - zfun: $(p.zfun)
   - Π : $(p.Π)")

distributions = [
LogNormal(x->0.0,exponentialcovariance(0.3,sqrt(0.1),2))
LogNormal(x->0.0,exponentialcovariance(0.3,sqrt(0.5),2))
LogNormal(x->0.0,exponentialcovariance(0.3,sqrt(0.01),2))
LogNormal(x->log(10.0),exponentialcovariance(0.3,sqrt(0.01),2))
LogNormal(x->log(10.0),exponentialcovariance(0.3,sqrt(0.001),2))
]

lim(u,lower,upper) = max.(min.(u,upper),lower)
blob(a,b) = x-> a<x<b ? 0.5-0.5*cos((x-a)*2π/(b-a)) : 0.0
blob2(a,b) = x-> a<x<b ? 2*(x-a)^2*(b-x)/(b-a)^3 : 0.0

problems1 = [
Problem1(1e-6, distributions[1], 0, 17, 3, blockfun)
Problem1(1e-6, distributions[1], 0, 17, 4, blockfun) #Paper condidate
Problem1(1e-6, distributions[2], 0, 9, 4, blockfun)
]

problems2 = [
Problem2(1e-1, distributions[1], 0, 9, 3, x->sin(π*x))
Problem2(1e-6, distributions[2], 0, 9, 3, x->sin(π*x))
Problem2(1e-6, LogNormal(x->0.0,exponentialcovariance(0.3,sqrt(0.001),2)), 0, 9, 3, x->sin(π*x))
Problem2(1e-6, LogNormal(x->0.0,exponentialcovariance(0.3,sqrt(0.1),2)), 0, 9, 5, x->sin(π*x)) #Paper candidate
Problem2(1e-6, LogNormal(x->0.0,exponentialcovariance(0.3,sqrt(0.1),2)), 0, 9, 5, blockfun)
]

problems3 = [
# ↓Problems3[1]↓
Problem3(0.0, distributions[3], -1.0, 1.0, 0, (33,501), (2,1), 2, x->0.1*sin(π*x))
Problem3(0.0, distributions[4], -1.0, 1.0, 0, (33,1001), (2,1), 2, x->0.6<x<0.8 ? 0.1 : 0.0)
Problem3(0.0, distributions[1], -1.0, 1.0, 0, (9,5000), (2,1), 3, x->sin(π*x))
Problem3(0.0, Deterministic(10), -1.0, 1.0, 0, (33,1001), (2,1), 2, x->0.25*blob(0.4,0.8)(x)) #relatively nice result
Problem3(0.0, Deterministic(1), -1.0, 1.0, 0, (9,10000), (2,1), 3, x->0.2(1-cos(2π*x))) # run this one for about 40 iterations as an example of an oscillating control as solution. (kscale = 1e-3)
Problem3(0.0, Deterministic(10), -1.0, 1.0, 0, (9,10000), (2,1), 3, x->0.1(1-cos(2π*x))) # Good problem, relatively smooth solution. (kscale = 1e-3)
Problem3(0.0, Deterministic(10), -1.0, 1.0, 0, (9,10000), (2,1), 5, x->0.15(1-cos(2π*x)))
Problem3(0.0, Deterministic(1), -1.0, 1.0, 0, (33,501), (2,1), 2, #=  x->0.3(1-cos(2π*x))  =# blob2(0.4,0.8),u->lim(u,0,1))
Problem3(0.0, Deterministic(5), -1.0, 1.0, 0, (33,1001), (2,1), 2, blob2(0.4,0.8)) #works with exp1
Problem3(1e-3, Deterministic(10), -1.0, 1.0, 0, (33,1001), (2,1), 2, x->0.25*blob(0.4,0.8)(x))
# ↓Problems3[11]↓
Problem3(1e-2, Deterministic(1), -1.0, 1.0, 0, (33,1001), (2,1), 2, x->0.25*blob(0.4,0.8)(x))
Problem3(2e-2, Deterministic(5e-1), -1.0, 1.0, 0, (33,1001), (2,1), 2, x->0.25*blob(0.4,0.8)(x)) #change 33 to higher to slower convergence speed of ncg finest level method.
Problem3(1e-3, Deterministic(10), -1.0, 1.0, 0, (33,1001), (2,1), 2, x->0.25*blob(0.4,0.8)(x))
Problem3(1e-3, distributions[4], -1.0, 1.0, 0, (33,1001), (2,1), 2, x->0.25*blob(0.4,0.8)(x))
Problem3(1e-3, distributions[4], -1.0, 1.0, 0, (9,1001), (2,1), 4, x->0.25*blob(0.4,0.8)(x))
Problem3(1e-3, distributions[5], -1.0, 1.0, 0, (33,10001), (2,1), 4, x->0.25*blob(0.4,0.8)(x))
]

end
