"""
    MLMC

The MLMC (multilevel Monte Carlo) module provides an implementation of the MLMC
estimator in the function [`mlmc`](@ref)

See also: [`mlmc`](@ref), [`SampleStats`](@ref), [`MLSampleStats`](@ref), [`nbsamples`](@ref), [`getn`](@ref), [`getϵ`](@ref), [`getVs`](@ref), [`RMSEType`](@ref), [`AggregateRMSE`](@ref), [`PointwiseRMSE`](@ref), [`mean`](@ref), [`var`](@ref), [`cov`](@ref)
"""
module MLMC
# This module groups all MLMC related methods
# It has no dependencies to make it as general as possible

import Base: getindex, setindex!, isassigned, size, length
import Statistics: mean, var, cov
using LinearAlgebra: norm
export 𝔼, 𝕍, ℂ, 𝕊
export mlmc, SampleStats, MLSampleStats, addsample!, nbsamples, getn, getϵ, getVs
export RMSEType, AggregateRMSE, PointwiseRMSE
export nb_solves

using Plotter

const DEBUG = false

𝔼 = mean
𝕍 = var
ℂ = cov
𝕊(args...) = sqrt.(𝕍(args...))

# single level sampling data for a single QoI of type T
# correlated samples are supported
"""
`SampleStats` provides single level sampling results for a single QoI of type T. T must implement methods that allow it to be interpreted as a vector space over the [`Real`](@ref) type, i.e., it must support addition, subtraction and scalar multiplication with a [`Real`](@ref), and `zero(::T)`. Furthermore, pointwise multiplication using `.*` must be supported (for the purpose of calculating variances) and also `sqrt.`.

    SampleStats(t, b::Int=0)

Constructs an **EMPTY** SampleStats{T} object, where T = typeof(t). b specifies that samples are allowed to be correlated with the b preceding samples. If b=0, all samples must be independent.
To add a sample, use [`addsample!`](@ref).

# Examples
```julia-repl
julia> samplestats = SampleStats(zeros(5))
MLMC.SampleStats{Array{Float64,1}}
```
```julia-repl
julia> addsample!(samplestats,[1.0,2.0,3.0,4.0,5.0])
MLMC.SampleStats{Array{Float64,1}}
```

Denote the full set of n samples added to a SampleStats{T} instance by s = [s[1],…,s[n]]. s is not fully stored. Only Σ(s[1],…,s[n]) and Σ(s[1].*s[1],…,s[n].*s[n]), Σ(s[2].*s[1],…,s[n].*s[n-1]], ..., Σ(s[b].*s[1],…,s[n].*s[n-b]] and the b starting samples s[1],…,s[b] and b ending samples s[n-b+1],…,s[n] are stored.

See also: [`addsample!`](@ref), [`MLSampleStats`](@ref), [`nbsamples`](@ref), [`mean`](@ref), [`var`](@ref), [`cov`](@ref)
"""
mutable struct SampleStats{T} # T is the type of the QoI
    n::Int #number of samples
    b::Int #correlation distance; 0 for no correlation, 1 for correlation with previous and next sample etc.
    sum::T #sum of all samples
    sumprod::Vector{<:T} #contains [nE[X_i^2], nE[X_iX_{i-1}], ..., nE[X_iX_{i-b}]], length is b+1
    # correlation distance is measured by bandwidth of sample covariance matrix. b=0 means all samples are independent
    first_samples::Vector{<:T} #must contain b first samples
    last_samples::Vector{<:T} #must contain b last samples
    function SampleStats{T}(z::T, b::Int = 0) where T # constructor, z must be of type T
        sum = zero(z)
        sumprod = [zero(z) for i in 0:b]
        first_samples = [zero(z) for i in 1:b]
        last_samples = [zero(z) for i in 1:b]
        new(0,b,sum,sumprod,first_samples,last_samples)
    end
end
SampleStats(z::T, b::Int = 0) where T = SampleStats{T}(z,b)

"""
    addsample!(::SampleStats{T}, sample::T)

Adds the given `sample` of type `T` to the provided `SampleStats{T}`.
"""
function addsample!(stats::SampleStats{T}, sample::T) where T
    stats.n+=1;
    stats.n <= stats.b && (stats.first_samples[stats.n] = sample)
    # Do updates for stats.last_sample[1] (or sample if stats.b==0)
    temp = [stats.last_samples; [sample]]
    stats.sum+=temp[1]
    stats.sumprod+=[temp[1].*t for t in temp]
    stats.last_samples = temp[2:end]
    return stats
end

"""
    mean(::SampleStats)

Computes the mean of all samples added to the [`SampleStats`](@ref) object using [`addsample!`](@ref).
"""
mean(stats::SampleStats) = stats.b==0 ? stats.sum/stats.n : (stats.sum+sum(stats.last_samples))/stats.n

"""
    cov(::SampleStats)

Computes the `b` covariances of all samples added to the [`SampleStats`](@ref) (see documentation for explanation of `b`) object using [`addsample!`](@ref).
"""
function cov(stats::SampleStats)
    #closing loop
    sumprod = stats.sumprod
    loop = [stats.last_samples; stats.first_samples]
    for i in 1:stats.b
        sumprod += [loop[i].*s for s in loop[i:i+stats.b]];
    end
    #calculating quantities
    sumprod./stats.n .- [mean(stats).^2]
end

"""
    var(::SampleStats)

Computes the variance of all samples added to the [`SampleStats`](@ref) object using [`addsample!`](@ref).
"""
var(stats::SampleStats) = cov(stats)[1] #NOTE: this is more efficient than you might expect.

# multilevel sampling data for a single QoI
"""
MLSampleStats stores sampling data for multiple levels, for a **single** QoI. In particular, it contains a `Vector{SampleStats}` for each level and the function mapping between the levels for the QoI in question.

See also: [`SampleStats`](@ref), [`ml_eval`](@ref), [`nbsamples`](@ref), [`mean`](@ref), [`var`](@ref), [`cov`](@ref)
"""
mutable struct MLSampleStats
    S::Vector{SampleStats}
    lm::Function # default mapping function
end
MLSampleStats(L::Int, lm::Function) = MLSampleStats(Vector{SampleStats}(undef,L), lm)
getindex(ML::MLSampleStats, args...) = getindex(ML.S, args...)
setindex!(ML::MLSampleStats, args...) = setindex!(ML.S, args...)
isassigned(ML::MLSampleStats, args...) = isassigned(ML.S, args...)
size(ML::MLSampleStats, args...) = size(ML.S,args...)
length(ML::MLSampleStats, args...) = length(ML.S,args...)

"""
    nbsamples(ss::SampleStats{T})::Int

Returns the number of samples added using [`addsample!`](@ref)

See also: [`SampleStats`](@ref), [`addsample!`](@ref)
"""
nbsamples(ss::SampleStats{T}) where T = ss.n

"""
    nbsamples(ML::MLSampleStats)::Vector{Int}

Returns the number of samples added on each of the levels using [`addsample!`](@ref)

See also: [`MLSampleStats`](@ref), [`SampleStats`](@ref), [`addsample!`](@ref)
"""
nbsamples(ML::MLSampleStats) = [nbsamples(ML.S[ℓ+1]) for ℓ in 0:length(ML)-1]

function ml_eval(ML::MLSampleStats, f::Function, lm::Function)
    S = ML.S
    fml = f(S[1]) #level ℓ=0
    for ℓ = 1:length(S)-1
        fℓ = f(S[ℓ+1]) #level ℓ
        fml = lm(fml,ℓ)+fℓ #mapping and adding
    end
    return fml
end
ml_eval(ML::MLSampleStats, f::Function) = ml_eval(ML, f, ML.lm)
mean(ML::MLSampleStats) = ml_eval(ML, mean)
var(ML::MLSampleStats) = ml_eval(ML, var)
cov(ML::MLSampleStats) = ml_eval(ML, cov, (vs,ℓ)->[ML.lm(vs[i],ℓ) for i in 1:length(vs)])

#### INFO ####
#inputs to the MLMC system:
# - a function that generates samples from the (PDE) problem. The MLMC module should have no knowledge of underlying problem
# - a deterministic part that should always be added at the end (important for a relative tolerance) (Not implemented)
# - a tolerance requirement on the RMSE, that is defined in some way.
# - a RMSE function (?) optional feature, otherwise, standard RMSE will be used.
##############

"""
    `RMSEType`

The valid `RMSEType`s are listed below:
- `PointwiseRMSE`: Defines the MSE as the pointwise mean square of the error
- `AggregateRMSE`: Defines the MSE as the mean square norm of the error
"""
abstract type RMSEType end
struct AggregateRMSE <: RMSEType end
struct PointwiseRMSE <: RMSEType end

"""
    mlmc(gensample, lm, b, n, nb_qoi::Int=1)

MLMC method implementation. The function `gensample` that generates samples of the Quantity of Interest (QoI) should have the following form:

    gensample(return_lvl, lvl, i, n)

with `(lvl::Int, i::Int)` fully identifying the sample and `return_lvl::Vector{Int}` the discretization level(s) at which the sample must be returned. `gensample` returns `nb_qoi` quantities of interest in a `nb_qoi`-Tuple (or in a `Vector` of such Tuples if `return_lvl` was a `Vector{Int}`). Functions to map each output of `gensample` generated for different values of `return_lvl` should be provided in `lm::Vector{Function}`, containing `nb_qoi` functions `lm(v,ℓ)` mapping `v` to level `ℓ`. For scalar QoI, one usually has `lm(v,ℓ) = v`. The correlation distances `b::Vector{Int}` contain, for each of the `nb_qoi` return values, the maximum difference in index `i` (for the same `lvl`) for which the samples may be correlated. The final sample indeces `n::Vector{Int}` can be used for wrapping around correlated samples.

If `nb_qoi==1`, one may also supply the parameters `lm` and `b` as `::Function` and `::Int` instead of as a `Vector` of length one of those.
"""
function mlmc(gensample::Function, lm::Vector{<:Function}, b::Vector{Int}, n::Vector{Int}, nb_qoi::Int=1)
    length(lm)==length(b)==nb_qoi || @error("lengths of lm ($(length(lm))) and b ($(length(b))) must be equal to nb_qoi ($nb_qoi)")
    all(n.>0) || @warn("At least a single sample will be taken on each level")
    n = max.(1,n)
    MLs = [MLSampleStats(length(n), lm[qoi]) for qoi in 1:nb_qoi]  #contains MLSampleStats for each QoI
    n_start = ones(Int,size(n))
    mlmc!(MLs, gensample, b, n_start, n, nb_qoi)
end
mlmc(gensample::Function, lm::Function, b::Int, n::Vector{Int}, nb_qoi::Int=1) = mlmc(gensample, fill(lm,nb_qoi), fill(b,nb_qoi), n, nb_qoi)

# NOTE: wrapping is probably bad, since mlmc! gets called for starting samples (don't need wrapping)
# NOTE: Probably best to discard wrapping altogether, check theory, and whether still gradient of something.

"""
    mlmc!(MLs::Vector{MLSampleStats}, gensample, b::Vector{Int}, n_start::Vector{Int}, n_end::Vector{Int}, nb_qoi::Int)

Same as `mlmc(gensample, lm, b, n, nb_qoi)`, but adds to a potentially already filled, `MLSampleStats` object. Note that `MLs` contains `lm`. The objects in `MLs` must be compatible with the provided `gensample` function.
n_start[ℓ] contains starting sample index for lvl ℓ
n_end[ℓ] contains last sample index for lvl ℓ
"""
function mlmc!(MLs::Vector{MLSampleStats}, gensample::Function, b::Vector{Int}, n_start::Vector{Int}, n_end::Vector{Int}, nb_qoi::Int)
    length(MLs)==nb_qoi || @error("length of MLs ($(length(MLs))) must be equal to nb_qoi ($nb_qoi)")
    length(n_start)==length(n_end) || @error("lengths of n_start ($(length(n_start))) and n_end ($(length(n_end))) must be equal")

    # Updates MLs at level lvl with sample s (all QoI)
    function updateMLs(s,lvl)
        for qoi in 1:nb_qoi
            addsample!(MLs[qoi][lvl+1], s[qoi])
        end
    end

    # puts input in a Tuple if it isn't already. A single QoI then behaves exactly the same as multiple QoIs.
    encapsulate(v::Tuple) = v
    encapsulate(v::Any) = (v,)

    # generates the difference between the same gensample samples on consecutive levels lvl and lvl-1 for all QoI
    function Δsample(lvl::Int,i::Int)
        if lvl==0
            s = encapsulate(gensample([lvl],lvl,i,n_end[lvl+1])[1])
        else
            s_fc = gensample([lvl,lvl-1], lvl, i, n_end[lvl+1]) #s_fc contains a vector of QoIs or tuples of QoIs.
            s_f = encapsulate(s_fc[1]); s_c = encapsulate(s_fc[2]);
            s = Tuple([s_f[q]-MLs[q].lm(s_c[q],lvl) for q in 1:nb_qoi]);
        end
        return s
    end

    for ℓ in 0:length(n_start)-1
        n_start[ℓ+1]<=n_end[ℓ+1] || continue # no samples taken on this lvl
        n_offset=0 #becomes 1 if the block below is executed
        if !isassigned(MLs[1], ℓ+1) # if no sample has ever been taken on this lvl
            all([!isassigned(MLs[qoi], ℓ+1) for qoi in 2:nb_qoi]) || @error("Internal error: A sample has been taken for one QoI but not for another QoI.")
            s = Δsample(ℓ,n_start[ℓ+1]) #first sample
            for qoi in 1:nb_qoi
                MLs[qoi][ℓ+1] = SampleStats(s[qoi],b[qoi])
            end
            updateMLs(s,ℓ)
            n_offset+=1
        end
        for i in n_start[ℓ+1]+n_offset:n_end[ℓ+1]
            s = Δsample(ℓ,i)
            updateMLs(s,ℓ)
        end
    end
    return MLs
end

"""
    mlmc(gensample, lm, b, ϵs, costs, n_max::Vector{Int}, nb_qoi::Int=1; rmsetype=PointwiseRMSE())

MLMC with tolerance requirement. Same as `mlmc(gensample, lm, b, n, nb_qoi)`, but determines `n` given the RMSE tolerances `ϵs` (`::Vector{<:Real}` with length `nb_qoi`) and the relative costs of a sample for each lvl `costs` (`::Vector{<:Real}` with length equal to the number of levels).
The tolerance is for the inf-norm of the RMSE if `rmsetype=PointwiseRMSE()`, and for the 2-norm if `rmsetype=AggregateRMSE()`.

If `nb_qoi==1`, one may also supply the parameters `lm`, `b`, and `ϵs` as `::Function`, `::Int`, and `::Real` instead of as a `Vector` of length one of those.
"""
function mlmc(gensample::Function, lm::Vector{<:Function}, b::Vector{Int}, ϵs::Vector{<:Real}, costs::Vector{<:Real}, n_max::Vector{Int}, nb_qoi::Int=1; rmsetype::RMSEType=PointwiseRMSE(), n_warmup::Vector{Int}=Int[])
    length(lm)==length(b)==length(ϵs)==nb_qoi || @error("lengths of lm ($(length(lm))), b ($(length(b))) and ϵs ($(length(ϵs))) must be equal to nb_qoi $nb_qoi")
    length(costs)==length(n_max) || @error("lengths of costs ($(length(costs))) and n_max ($(length(n_max))) must be equal (to number of levels)")
    nb_ℓ = length(n_max) # number of levels (fixed at start)

    costs = Cs(costs) # takes into account that each sample on level ℓ requires a call to gensample on level ℓ-1 also.

    #do a number of warmup samples
    if n_warmup == []
        # n₀_warmup = 250; n_warmup_min = 5; warmup_decay = 2
        # all(n_warmup_min.<=n_max) || @warn("The maximum allowed number of samples is so small that error estimation might be inaccurate.")
        # n_warmup = [min(n_max[i], max(n_warmup_min, ceil(Int,n₀_warmup/warmup_decay^(i-1))) ) for i=1:nb_ℓ]
        n_warmup_max = 2500
        n_warmup_finest = 4
        n_warmup = min.(n_warmup_max,n_max,ceil.(Int,(n_warmup_finest*costs[end])./costs))
    end
    MLs = mlmc(gensample, lm, b, n_warmup, nb_qoi)

    #decide how many samples will be needed
    n = Matrix{Int}(undef,nb_qoi,nb_ℓ)
    for q in 1:nb_qoi
        ϵ = ϵs[q]
        if isinf(ϵ)
            n[q,:].=0 #no samples have to be taken for that QoI
            continue
        end
        n[q,:] = getn(nb_ℓ-1,getVs(MLs[q],rmsetype),costs,ϵ,rmsetype)
    end
    DEBUG && println(n)
    n = vec(maximum(n,dims=1)) # collapses first dimension, i.e., the QoI dimension
    all(n.<=n_max) || @warn("Number of required samples exceeds maximum allowed number of samples. The requested tolerance might not be satisfied.\nRequired number of samples: $n\nMaximum number of samples: $n_max")

    #take additional samples, if necessary
    MLs = mlmc!(MLs, gensample, b, n_warmup.+1, n, nb_qoi)
end
mlmc(gensample::Function, lm::Function, b::Int, ϵ::Real, costs::Vector{<:Real}, n_max::Vector{Int}, nb_qoi::Int=1; rmsetype::RMSEType=PointwiseRMSE()) = mlmc(gensample, fill(lm,nb_qoi), fill(b,nb_qoi), fill(ϵ,nb_qoi), costs, n_max, nb_qoi; rmsetype=rmsetype)

# getVs: returns relevant variances and covariances on finest level
# Also takes covariances into account, despite the name
function getVs(ML::MLSampleStats, ::PointwiseRMSE)
    nb_ℓ = length(ML)
    Vs = [sum( cov(ML[ℓ]).*[1.0; 2.0.*ones(ML[ℓ].b)] ) for ℓ in 1:nb_ℓ]
    all(minimum.(Vs).>= -1e-10) || @warn("getVs: minimum variance is $(minimum.(Vs))!, b=$(ML[1].b)")
    Vs = [max.(v,0) for v in Vs]
    if DEBUG
        for ℓ in 1:nb_ℓ
                Plotter.figure(ℓ);
                Plotter.surf(Vs[ℓ])
                println("getVs: ℓ = $(ℓ-1), Vs[ℓ+1] is bounded by $(minimum(Vs[ℓ])) and $(maximum(Vs[ℓ]))");
        end
    end
    Vs = [ML.lm(Vs[ℓ],nb_ℓ-1) for ℓ in 1:nb_ℓ] # get all Vs on finest level.
    return Vs
end
function getVs(ML::MLSampleStats, ::AggregateRMSE)
    nb_ℓ = length(ML)
    # NOTE: also takes covariances into account, despite the name
    Vs = [sum( cov(ML[ℓ]).*[1.0; 2.0.*ones(ML[ℓ].b)] ) for ℓ in 1:nb_ℓ]
    all(minimum.(Vs).>= -1e-10) || @warn("getVs: minimum variance is $(minimum.(Vs))!")
    c(::Any) = 1; c(v::Array) = length(v) #NOTE: might want to allow custom norm as a field of AggregateRMSE
    Vs = [norm(v,1)/c(v) for v in Vs]
    DEBUG && Plotter.semilogy(0:(nb_ℓ-1), Vs)
    return Vs
end

# getn: determine optimal sample sizes for a single QoI
# L is finest level
# Vs is Vector of variances, on all levels, mapped to the finest level L
# Cs is a vector containing relative cost of a sample for each lvl
# ϵ is the tolerance on the absolute RMSE
function getn(L::Int,Vs::Vector,Cs::Vector{<:Real},ϵ::Real,::PointwiseRMSE;Cϵ::Real=1.0,TOLscale::Real=1.0)
    # each nvec[ℓ+1] will contain the amount of samples required at lvl ℓ
    # for each point of the QoI. The infnorm yields the sample size.
    # if the distribution of n is different for the different elements of the QoI
    # it might be beneficial to do adaptive refinement of the mesh instead.
    nvec = fill(zero(Vs[1]), length(Vs))
    Σsqrt = zero(Vs[1])
    for ℓ = 0:L
        Σsqrt += sqrt.(Vs[ℓ+1]*Cs[ℓ+1])
    end
    for ℓ = 0:L
        #DEBUG && println("getn: V[$ℓ] = $(maximum(Vs[ℓ+1])); C[$ℓ] = $(Cs[ℓ+1]) and n is proportional to ", sqrt(maximum(Vs[ℓ+1])/Cs[ℓ+1]))
        nvec[ℓ+1] = (Cϵ*TOLscale/ϵ)^2 .*sqrt.(Vs[ℓ+1]/Cs[ℓ+1]).*Σsqrt #nvec contains real values
    end
    n = [ceil(maximum(nvec[ℓ+1])) for ℓ in 0:L]
end
function getn(L::Int,Vs::Vector,Cs::Vector{<:Real},ϵ::Real,::AggregateRMSE;Cϵ::Real=1.0,TOLscale::Real=1.0)
    nvec = fill(zero(Vs[1]), length(Vs))
    Σsqrt = sum(sqrt.(Vs.*Cs))
    n = ceil.( (Cϵ*TOLscale/ϵ)^2 .*Σsqrt.*sqrt.(Vs./Cs) )
end

# NOTE: It might be necessary to take a norm of the result, since it is not necessarily a number, but rather of the same type as the QoI
"""
    getϵ(::MLSampleStats, ::RMSEType)

calculates the expected absolute RMSE, whose definition is decided by the [`RMSEType`](@ref) provided.
"""
function getϵ(ML::MLSampleStats, rmsetype::RMSEType)
    nb_ℓ = length(ML)
    Vs = getVs(ML, rmsetype) # calculate relevant variances and covariances
    if DEBUG
        for ℓ = 0:nb_ℓ-1
            println("getϵ: [$ℓ] = $(ML[ℓ+1].n); maxV = $(maximum(Vs[ℓ+1]))")
            println("getϵ: ϵ contribution on $ℓ = ", 1/float(ML[ℓ+1].n) * maximum(Vs[ℓ+1]))
        end
    end
    sqrt.(sum([Vs[ℓ]/ML[ℓ].n for ℓ in 1:nb_ℓ])) # calculate ϵ
end

"""
    Cs(costs::Vector{<:Real})

Generates Cs, where Cs[i] = costs[i]-costs[i-1], and Cs[1] = costs[1]. Cs[i] then gives the cost for a MLMC sample at level i.
"""
Cs(costs::Vector{<:Real}) = [costs[1]; [costs[i]+costs[i-1] for i=2:length(costs)]]

"""
    nb_solves(n::Vector{Int}, costs::Vector{<:Real})

Calculates the equivalent number of finest level solves from the number of samples on each level `n` and the `costs` for a solve on each level.
"""
nb_solves(n::Vector{Int},costs::Vector{<:Real}) = sum(Cs(costs).*n)/costs[end]

end
