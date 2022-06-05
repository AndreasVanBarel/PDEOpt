"""
    Gradient

The Gradient module takes a function that generates a deterministic gradient and constructs a function providing samples of costfunction and gradient compatible with [`MLMC.mlmc`](@ref). The MLMC results are interpreted and returned in a clean way to the user.

See also: [`SamplingData`](@ref), [`ComputeStruct`](@ref)
"""
module Gradient

using MLMC

import Base.show
export SamplingData, ComputeStruct, construct_gensample

"""
    SamplingData

Stores the sampling data, consisting of
  * `seed` - a seed
  * `n`    - A Vector containing the number of samples to be taken on each MLMC level.
"""
struct SamplingData
    seed::Int
    n::Vector{Int}
end

"""
    ComputeStruct

Stores all problem specific data into a callable struct containing
  * `cost_grad_state` - Function taking `(u,k)` returning `(J,∇J,y)`.
  * `k_sampler_generator`     - takes a `seed` and returns a `k_sampler` (see below).
  * `restrict_k`      - Function restricting parameters `k` to a coarser level.
  * `lm_u`            - mapping function for control `u`.
  * `lm_y`            - mapping function for `y`.
  * `lm_∇J`           - mapping function for `∇J`.
  * `L`               - maximum level.
  * `costs::Vector{Float64}`  - cost of a sample at each level.
  * `rmsetype = PointwiseRMSE()`  - The `RMSEType` to be used, see [`MLMC.RMSEType`](@ref).

`k_sampler`     - Function taking (level `ℓ`, sample `i`), returning, discretized at level `ℓ`, the parameter sample indexed by `(ℓ,i)`.

`restrict_k`    - Function taking (parameters `k`, level `ℓ`, return level `rℓ`), where `k` is given on level `rℓ` and `rℓ≤ℓ`, and returning `k` restricted to level `rℓ`.

The other mapping functions are of the form\n
`lm_*(v0,v1)` maps `v0` to the level of `v1` (`v1` is overwritten).
"""
struct ComputeStruct <: Function
    cost_grad_state::Function #takes control u and provides (J,∇J,y)
    k_sampler_generator::Function #takes a seed and returns a k_sampler
    restrict_k::Function #mapping function for parameters k
    lm_u::Function #mapping function for control u
    lm_y::Function #mapping function for y
    lm_∇J::Function #mapping function for ∇J
    L::Int #maximum level
    costs::Vector{Float64} #TODO incorporate (and estimate) costs in MLMC itself
    rmsetype::RMSEType
end
ComputeStruct(cgs,k,r_k,lm_u,lm_y,lm_∇J,L,costs) = ComputeStruct(cgs,k,r_k,lm_u,lm_y,lm_∇J,L,costs,PointwiseRMSE())
show(io::IO, ::MIME"text/plain", c::ComputeStruct) = print(io, "ComputeStruct containing levels 0..$(c.L). See ?ComputeStruct.")

"""
    (::ComputeStruct)(u, s::SamplingData)

  * `u` - control input at the finest level.
  * `s` - contains sampling data.
Returns cost `J`, gradient `∇J`, expected state `𝔼y`, state variance `𝕍y`, estimated RMSE `ϵ` (as defined by `rmsetype`).
"""
function (g::ComputeStruct)(u, s::SamplingData)
    us = [g.lm_u(u,ℓ) for ℓ in 0:g.L] # map u to all relevant levels.
    gensample = construct_gensample(g,s.seed) #g.cost_grad_state, param_sampler, g.lms, g.lm_k)
    gensample_us = (rlvls, lvl, i, n)->gensample(us, rlvls, lvl, i, n)
    result = mlmc(gensample_us,[(x,ℓ)->x, g.lm_∇J, g.lm_y],[0,0,0],s.n,3)
    unwrap(result)..., maximum(getϵ(result[2], g.rmsetype)) # returns J,∇J,𝔼[y],𝕍[y],ϵ
end

"""
    (::ComputeStruct)(u, ϵ[, seed::Int])

  * `u` - control input at the finest level.
  * `ϵ` - Required RMSE on gradient.
Returns cost `J`, gradient `∇J`, expected state `𝔼y`, state variance `𝕍y`, SamplingData `s`.
"""
function (g::ComputeStruct)(u, ϵ::Real, seed::Int=abs(rand(Int)))
    n_maxL = 1000 #maximum number of expensive samples.
    us = [g.lm_u(u,ℓ) for ℓ in 0:g.L] # map u to all relevant levels.
    gensample = construct_gensample(g,seed)
    gensample_us = (rlvls, lvl, i, n)->gensample(us, rlvls, lvl, i, n)
    ϵs = [Inf,ϵ,Inf] #only gradient needs to be precise.
    n_max = Int.(ceil.(g.costs[end]./g.costs.*n_maxL))
    result = mlmc(gensample_us,[(x,ℓ)->x, g.lm_∇J, g.lm_y],[0,0,0],ϵs,g.costs,n_max,3; rmsetype=g.rmsetype)
    n = nbsamples(result[1])
    s = SamplingData(seed,n)
    unwrap(result)..., s # returns J, ∇J, 𝔼y, 𝕍y, s
end

# unwrap: unwraps output from MLMC.mlmc(…)
function unwrap(MLs::Vector{MLSampleStats})
    J_stat = MLs[1]
    ∇J_stat = MLs[2]
    y_stat = MLs[3]
    J = 𝔼(J_stat)
    ∇J = 𝔼(∇J_stat)
    𝔼y = 𝔼(y_stat)
    𝕍y = 𝕍(y_stat)
    return J, ∇J, 𝔼y, 𝕍y
end

# Saves stuff for performance reasons
mutable struct Cache
    V::Vector{Tuple{Any,Any}}
    d::Int #number of locatons filled, i.e., V[1:d] is filled
    Cache(n::Int) = new(Vector{Tuple{Any,Any}}(undef,n),0)
end
iscached(c::Cache, key) = any([c.V[i][1]==key for i in length(c.V)-c.d+1:length(c.V)])
function cache(c::Cache,key,data)
    if c.d == length(c.V)
        c.V = [(key,data); c.V[1:end-1]]
    else
        c.V[end-c.d] = (key,data)
        c.d+=1
    end
end
function retrieve(c::Cache, key)
    for i in length(c.V):-1:length(c.V)-c.d+1
        key == c.V[i][1] && return c.V[i][2]
    end
    @error("Nothing was found in the cache. Run iscached(...) first.")
end

"""
    construct_gensample(g::ComputeStruct, seed::Int)

constructs gensample. See also [`ComputeStruct`](@ref).

`gensample(us::Vector, rℓs::AbstractVector{Int}, ℓ::Int, i::Int)`
   `us`      - Vector of control inputs at all levels.
   `rℓs`     - Vector of m ≥ 1 return levels.
   `ℓ`       - Level at which the parameters should be sampled. Assumed that all(ℓ.<=rℓs).
   `i`       - Index of the sample.
returns
   A Vector of Tuples (J(uᵢ; kᵢ), ∇J(uᵢ; kᵢ), y(uᵢ; kᵢ)) of length m. Tuple i is calculated and returned at level rℓs[i]. It contains the result for control uᵢ = us[rℓs[i]] with parameter kᵢ generated by the given seed at index (ℓ,i) and mapped to rℓs[i] by g.lm_k. These Tuples in the Vector should be highly correlated. """
function construct_gensample(g::ComputeStruct, seed::Int)
    # unpack
    k_sampler = g.k_sampler_generator(seed) #generates the parameter sampler.
    cost_grad_state = g.cost_grad_state
    restrict_k = g.restrict_k

    k_cache = Cache(1) #stores the last taken sample of the parameters.

    # gets the (ℓ,i)-th parameter sample returned on level rℓ≤ℓ
    function get_k(rℓ::Int,ℓ::Int,i::Int)
        if iscached(k_cache, (ℓ,i))
            k = retrieve(k_cache, (ℓ,i))
        else
            k = k_sampler(ℓ,i)
            cache(k_cache, (ℓ,i), k)
        end
        restrict_k(k, ℓ, rℓ)
    end

    # sample: returns sample of J, ∇J, and y at lvl rℓ for sample (ℓ,i).
    # u must be provided at level rℓ
    function sample(rℓ::Int, ℓ::Int, i::Int, u)
        k = get_k(rℓ,ℓ,i)
        J, ∇J, y = cost_grad_state(u,k)
        return J, ∇J, y
    end

    function gensample(us::Vector, rℓs::AbstractVector{Int}, ℓ::Int, i::Int, n::Int=typemax(Int))
        all(rℓs.<=ℓ) || @error("rℓs[i]≤ℓ must hold ∀i.")
        all(rℓs.<=length(us)+1) || @error("It is necessary to provide the control on all levels ≤ ℓ.")

        samples = Vector{Any}(undef,length(rℓs)) # generate empty QoI vector
        for r in 1:length(rℓs)
            rℓ = rℓs[r]
            samples[r] = sample(rℓ, ℓ, i, us[rℓ+1])
        end
        return samples
    end

    return gensample
end

end
