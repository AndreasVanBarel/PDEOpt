module Optimization

using General
using MLMC
using Gradient
using Problems
using Printf
using LinearAlgebra
import Plotter
pp = Plotter

export getdir, quadmin, armijo, linesearch
export ncg
export gen_f, mgoptV, mgoptW, smooth

#############################
## Essential functionality ##
#############################
# Produces Vector{Pair} containing only those s=>options.s for the s in keywords that have a corresponding options.s.
function extractKwargs(object, keywords::Vector{Symbol})
    props = propertynames(object)
    kwargs = Pair[]
    for s in keywords
        s in props && push!(kwargs, s=>getproperty(object,s))
    end
    return kwargs
end

####################
## Line Searching ##
####################
"""
    getdir(∇J, ∇J₀, d₀)

Produces a search direction using the Dai-Yuan formula given current gradient ∇J, previous gradient ∇J₀ and previous search direction d₀."""
function getdir(∇J, ∇J₀, d₀)
    Δg = ∇J-∇J₀
    if norm(Δg)/norm(∇J)<1e-8 # may happen if step size is very small
        # @warn("extreme values detected in getdir. Returning -∇J.")
        return -∇J
    else
        β = norm(∇J)^2/(d₀ ⋅ Δg) # Dai-Yuan formula
        if isnan(β) # could happens if step size was zero. Normally should never happen.
            error("NaN detected for β. Returning -∇J.")
            return -∇J
        end
    end
    return -∇J+β*d₀
end

"""
    quadmin(u, g, d, ∇J; s_guess=1.0, s_min=-Inf, s_max=Inf, print=0)

Finds s such that ∇J(u+s*d) = 0.
g must be equal to ∇J(u) and is very likely calculated beforehand. This argument exists such that it does not have to be recalculated inside this function.

Constructs a linear model ∇Jm of the provided gradient ∇J by interpolating the given g=∇J(u) and g₁ = ∇J(u+s_guess*d). If ∇J:u↦∇J(u) provides the gradient of some quadratic cost function J, i.e., if ∇J is linear in u, then this model ∇Jm is exactly ∇J.

Calculates the step size s₀ such that ∇Jm(u+s₀*d)=0. Let s be the number in [s_min, s_max] closest to s₀.
`quadmin` returns the step size s, and the model gradient ∇Jm(u+s*d). Note that ∇Jm(u+s*d) can be calculated from the two interpolation points g and g₁ as s*g₁+(1-s)*g. The gradient function ∇J is therefore only called once in total. If ∇J is the gradient of some non-quadratic J, the resulting ∇Jm(u+s*d) can be considered a first estimate of ∇J(u+s*d). Setting print larger than 0 provides more information during the calculation.

For convenience, one may put all, or a subset of, the named arguments above into a struct or named tuple instead.
"""
function quadmin(u, g, d, ∇J; s_guess::Real=1.0, s_min::Real=-Inf, s_max::Real=Inf, print::Int=0)
    u₁ = u+s_guess*d
    g₁ = ∇J(u₁)
    c = g ⋅ d # directional derivative if norm(d) = 1
    c₁ = g₁ ⋅ d # directional derivative if norm(d) = 1
    s = c/(c-c₁) # c+s*(c₁-c) = 0
    s = max(s_min,min(s_max, s))
    g_new = s*g₁ + (1-s)*g
    return s, g_new
end
quadmin(u, g, d, ∇J, options; kwargs...) = quadmin(u, g, d, ∇J; extractKwargs(options,[:s_guess,:s_max,:print])..., kwargs...)

"""
    armijo(u, J, g, d, J∇J; s_guess=1.0, s_decay=0.25, β=0.1, max_evals=10, inner=dot, print=0)

Finds s such that Armijo's sufficient descent condition is satisfied, i.e., finds a value of s such that

    J∇J(u + s*d) < J + β*s*(g ⋅ d).

J∇J is a function returning the cost functional value as well as the gradient for a given control. J,g must be equal to J∇J(u) and are very likely calculated beforehand. These arguments exists such that they do not have to be recalculated inside this function. The dot-product is calculated using the provided Function inner.

Returns the step size s and the costfunction and gradient as given by J∇J(u+s*d). If none of the tested values for s results in sufficient descent, returns s=0.0 and J,g (which should equal J∇J(u+0*d)). The function J∇J is called at least once, and at most max_evals times. The first guess for s is given by s_guess, and subsequent guesses are obtained by multiplying with s_decay each time. Setting print larger than 0 provides more information during the calculation.

For convenience, one may put all, or a subset of, the named arguments above into a struct or named tuple instead.
"""
function armijo(u, J, g, d, J∇J; s_guess::Real=1.0, s_decay::Real=0.25, β::Float64=0.25, max_evals::Int=10, inner::Function=dot, print::Int=0)
    s_guess <= 0.0 && print>0 && println("s_guess ≤ 0, returning 0.0,J,g")
    s_guess <= 0.0 && return 0.0,J,g
    ⋅ = inner
    s = s_guess
    local J_new, g_new

    for i = 1:max_evals
        J_new,g_new = J∇J(u+s*d)
        # println("i = $i; s = $s; Δ = $(J_new - (J + β*s*(g ⋅ d))); $(norm(g_new))")
        # check for sufficient descent (Armijo's condition)
        if J_new < J + β*s*(g ⋅ d)
            print>0 && println("i = $i; s = $s; J(u+s*d) = $J_new < $(J + β*s*(g ⋅ d)) = J+β*s*(g⋅d)")
            return s, J_new, g_new
        else
            s*=s_decay
        end
    end

    print>0 && println("No sufficient descent after $max_evals tests of Armijo's condition.
    J(u+s*d) = $J_new > $(J + β*s*(g ⋅ d))!
    Returning step size 0.")
    print>1 && println("u=$u, g=$g, d=$d")
    return 0.0,J_new,g_new
end
armijo(u, J, g, d, J∇J, options; kwargs...) = armijo(u, J, g, d, J∇J; extractKwargs(options,[:s_guess,:s_decay,:β,:max_evals,:inner,:print])..., kwargs...)

"""
    linesearch(u, J, g, d, J∇J, options)

Performs a linesearch with the method given as a string in options.method. "quadmin" leads to [`quadmin`](@ref), "armijo" to [`armijo`](@ref). The options are passed to those methods. If options.method is "constant", then the linesearch step s will always result in s=options.stepsize. Attempts to return values s, J_new, g_new if they can be inferred from the linesearch method without any additional calls of J∇J. E.g., J_new is only known when using [`armijo`](@ref) and will be `nothing` when using [`quadmin`](@ref).
"""
function linesearch(u,J,g,d,J∇J,options)
    # if :u_norm_bound in propertynames(options)
    #     :u_norm in propertynames(options) || error("if options.u_norm_bound is specified, options.norm_u must also be specified")
    #     s_bound = (max_u_norm - options.norm_u(u))/options.norm_u(d)
    #     # s_bound >= 0 || @warn("Inconsistency during linesearch. The provided u to start from has a larger options.norm_u(u) than options.max_u_norm.")
    #     # s_bound = min(0,s_bound)
    # end
    getd(object,symbol,default) = symbol in propertynames(object) ? getproperty(object,symbol) : default
    s_bound = ( (:s_bound in propertynames(options)) ? options.s_bound(u,d) : Inf)::Float64
    if options.method=="armijo" #TODO make this also work if options.s_guess is not defined.
        if getd(options,:s_guess,1.0) > s_bound
            s,J_new,g_new = armijo(u,J,g,d,J∇J,options,s_guess=s_bound)
        else
            s,J_new,g_new = armijo(u,J,g,d,J∇J,options)
        end
        return s,J_new,g_new
    elseif options.method=="quadmin"
        ∇J(u) = J∇J(u)[2]
        if getd(options,:s_guess,1.0) > s_bound
            s,g_new = quadmin(u,g,d,∇J,options,s_guess=s_bound)
        else
            s,g_new = quadmin(u,g,d,∇J,options)
        end
        return s,nothing,g_new
    elseif option.method=="constant"
        # s_bound < linesearch.stepsize || @warn("The constant stepsize results in options.norm_u(u+s*d) > options.norm_u_bound.")
        return min(linesearch.stepsize,s_bound),nothing,nothing
    else
        error("Linesearch method $(options.method) is unknown.")
    end
end

#########################################
## Nonlinear conjugate gradient method ##
#########################################
#TODO: Consistency in the objects generated by the save option.
"""
    ncg(u, f, τ, it; norm, print, plotfun, save, ls_options, Π)

NCG method implementation.
# Arguments
- `u`: initial control estimate.
- `f::Function`: function u↦(J(u),∇J(u)).
- `τ`: tolerance on gradient norm.
- `it`: maximum number of iterations.

# Additional optional arguments
- `norm::Function=norm`: The norm to use when reporting ‖u‖ or ‖∇J‖
- `print::Int=1`: the amount of information to be printed, with `0` being nothing.
- `plotfun::Function`: if set, `plotfun(k,u,∇J,d)` is called after every iteration `k`, where `u` is the current iterate, `∇J` the current gradient and `d` the current search direction. Can be used to plot these quantities during the optimization.
- `save`: provide an empty `Array{Any}` in which iteration information will be stored as tuples `(u,J,∇J,d)`. The final value of `u` is not stored (but returned).
- `ls_options`: how the step size is determined. A named tuple
    `(method="quadmin",)` (default): Recommended for linear problems
    `(method="armijo",inner,...)`: Recommended for nonlinear problems
        inner should be a function providing the inner product w.r.t. which the gradient is defined. For more options, see [`armijo`](@ref).
- `Π::Function`: If set, after every linesearch, the current iterate u is projected onto the feasible set by `Π(u)`

# Returns
- Final iterate `u`
"""
function ncg(u, f::Function, τ::Real, it::Int; norm::Function=norm, print::Int=1, plotfun::Function=(x...)->nothing, save=false, ls_options=(method="quadmin",), Π=x->x)
    print>0 && println("NCG started.")
    local ∇J₀,d # Necessary because each for loop iteration allocates new variables.
    J, ∇J = f(u)
    n∇J = norm(∇J)
    print>1 && println("i    J              ‖∇J‖          last step")
    print>1 && @printf("%-3i  %12.10f  %8.6e  (none)\n", 0, J, n∇J)

    for k = 1:it
        n∇J<τ && break
        d = k==1 ? -∇J : getdir(∇J,∇J₀,d) # compute the new search direction
        s,_,_ = linesearch(u,J,∇J,d,f,ls_options)
        u += s*d
        u = Π(u)
        ∇J₀ = ∇J
        J, ∇J = f(u)
        n∇J = norm(∇J)

        # print, plot, save
        print>1 && @printf("%-3i  %12.10f  %8.6e  %8.2f\n", k, J, n∇J, s)
        plotfun(k,u,∇J,d)
        save!=false && push!(save,(u,J,∇J,d,s))
    end

    print>0 && (n∇J<τ ? println("NCG successful. Tolerance $τ reached ($n∇J).") : println("NCG has finished its maximum of $it iterations. Tolerance $τ not reached ($n∇J)."))
    return u
end

"""
    ncg(u, f, τ, it, s; norm, plotfun, print, save, ls_options, Π)

NCG method where `s` is used to specify how `f` should be sampled or otherwise calculated. The provided `f::Function` should behave as follows: `f(u,s)`↦`(J,∇J,𝔼y,𝕍y,ϵ)`. The other arguments have the same meaning as before except for the following:
- `plotfun::Function`: if set, `plotfun(k,u,J,∇J,𝔼y,𝕍y,d)` is called after every iteration `k`, where `u`, `J`, `∇J` and `d` are as before and `𝔼y` and `𝕍y` come from the provided `f`.
- `save`: provide an empty `Array{Any}` in which iteration information will be stored as tuples `(k,u,J,∇J,𝔼y,𝕍y,d,ϵ,s)`. The final value of `u` is again not stored (but returned)."""
function ncg(u, f::Function, τ::Real, iterations::Int, samplingparams; norm::Function=norm, plotfun::Function=(x...)->nothing, print::Int=1, save=false, ls_options=(method="quadmin",), Π=x->x)
    print>0 && println("NCG using fixed samples started.")
    print>1 && println("i    J             ϵ           ‖∇J‖          last step")
    ∇J₀ = zero(u)
    local d # Necessary because each for loop iteration allocates new variables.
    for k = 1:iterations
        J, ∇J, 𝔼y, 𝕍y, ϵ = f(u,samplingparams)
        n∇J = norm(∇J)
        n∇J<τ && break
        d = k==1 ? -∇J : getdir(∇J,∇J₀,d) # compute the new search direction
        #println("getdir($(norm(∇J)),$(norm(∇J₀)), d) = $(norm(d))")
        J∇J = u->f(u,samplingparams)[1:2]
        s,_,_ = linesearch(u,J,∇J,d,J∇J,ls_options)
        plotfun(k,u,J,∇J,𝔼y,𝕍y,d)
        if save!=false
            push!(save,(k,u,J,∇J,𝔼y,𝕍y,d,s,ϵ,samplingparams))
        end
        print>1 && @printf("%-3i  %12.10f  %6.4e  %8.6e  %8.2f\n", k, J, ϵ, n∇J, s)

        #@printf("iteration %3i. J = %12.10f. ϵ = %6.4e. ‖∇J‖ = %8.6e. s = %8.2f.\n", k, J, ϵ, n∇J, s)

        u += s*d
        u = Π(u)
        ∇J₀ = ∇J
    end
    return u
end

"""
    ncg(u, f, τ, it, q::Real; norm, plotfun, print, save, ls_options, Π)

NCG method taking into account the accuracy of the gradient. `q` provides the bound for the relative RMSE on the gradient. The provided function `f` must provide 2 methods:
- `f(u,s)`↦`(J,∇J,𝔼y,𝕍y,ϵ::Real)` to obtain (amongst other things) the RMSE `ϵ` given the sampling method `s`.
- `f(u,ϵ::Real)`↦`(J,∇J,𝔼y,𝕍y,s)` to obtain (amongst other things) a sampling method `s` that attains a RMSE of at most `ϵ`.
Here `s` can obviously not be subtype of `Real`. The additional arguments have the same meaning as before."""
function ncg(u, f::Function, τ::Real, iterations::Int, q::Real; norm::Function=norm, print::Int=1, plotfun::Function=(x...)->nothing, save=false, ls_options=(method="quadmin",),Π=x->x)
    print>0 && println("NCG started.")
    local d, ∇J₀ # Necessary because each for loop iteration allocates new variables.
    ϵ = 0.01 # first requested ϵ
    η = 0.25 # decay of requested ϵ
    #q = 1.0 # bound for the relative RMSE on gradient
    J, ∇J, 𝔼y, 𝕍y, samplingdata = f(u,ϵ);
    save!=false && push!(save,(u,J,∇J,𝔼y,𝕍y,nothing,nothing,ϵ,samplingdata,0.0))
    print>1 && println("New n = $(samplingdata.n), for requested ϵ = $(@sprintf("%6.4e", ϵ)).")
    print>1 && println("i    J             ϵ           ‖∇J‖          last step")
    for k = 1:iterations
        tstart = time();
        if norm(∇J)<=τ
            print>1 && @printf("Fixed sample gradient is %8.6e. Testing ‖∇J‖ for convergence using new samples", norm(∇J))
            tested = norm(f(u,ϵ)[2])
            print>1 && @printf(": %8.6e.\n", tested)
            if tested <=τ
                return u
            end
        end
        d = k==1 ? -∇J : getdir(∇J,∇J₀,d) # compute the new search direction
        J∇J = u->f(u,samplingdata)[1:2]
        s,_,_ = linesearch(u,J,∇J,d,J∇J,ls_options)
        u += s*d
        u = Π(u)
        ∇J₀ = ∇J
        n∇J = norm(∇J)
        if ϵ > max(q*τ,q*n∇J) || ϵ < η^2*q*n∇J
            print>1 && @printf("Change of sample set (ϵ = %6.4e, ‖∇J‖ = %8.6e).\n", ϵ, n∇J)
            ϵ = max(q*τ,η*q*n∇J)
            J, ∇J, 𝔼y, 𝕍y, samplingdata = f(u,ϵ);
            print>1 && Base.print("New n = $(samplingdata.n), for requested ϵ = $(@sprintf("%6.4e", ϵ)).\n")
        else
            J, ∇J, 𝔼y, 𝕍y, ϵ = f(u,samplingdata)
        end
        print>1 && @printf("%-3i  %12.10f  %6.4e  %8.6e  %8.2f\n", k, J, ϵ, n∇J, s)

        plotfun(k,u,J,∇J,𝔼y,𝕍y,d)
        save!=false && push!(save,(u,J,∇J,𝔼y,𝕍y,d,s,ϵ,samplingdata,time()-tstart))
    end
    return u
end

############
## MG/OPT ##
############

"""
    gen_f(cs::Vector{ComputeStruct}, ss::Vector{SamplingData})

Generates function returning (J,∇J) for a given Vector of ComputeStruct and a Vector of SamplingData (of which elements k contains the ComputeStruct and SamplingData for MG/OPT level k)."""
function gen_f(cs::Vector{ComputeStruct}, ss::Vector{SamplingData})
    function f(u,ℓ)
        c = cs[ℓ+1]
        s = ss[ℓ+1]
        c(u,s)[1:2]
    end
end

"""
    smooth(u₀,f,μ[;breakcond,ls_options])

NCG smoother using μ smoothing steps. Returns new control iterate u, gradient in u, gradient in u₀ and number of NCG steps that were performed (equal to μ if no breakcond is set (see below)).

# Arguments
- `u`: current control iterate.
- `f::Function`: function u↦(J(u),∇J(u)).
- `μ::Int`: number of smoothing steps.

# Additional arguments
- `breakcond::Function`: breaks the smoothing once breakcond(∇J(u)) is satisfied.
- `ls_options`: how the step size is determined. A named tuple
    (method="quadmin",) (default): Recommended for linear problems
    (method="armijo",inner,...): Recommended for nonlinear problems
        inner should be a function providing the inner product w.r.t. which the gradient is defined. For more options, see [`armijo`](@ref).
"""
function smooth(u,f::Function,μ::Int;print=1,norm_∇J=norm,breakcond=g->false,ls_options=(method="quadmin",),ls_options_fallback=(method="none",))
    local d # Necessary because each for loop iteration allocates new variables.
    #if !state; ∇J_prev,d = state; end
    i = 0
    J,∇J = f(u)
    J₀,∇J₀ = J,∇J # save starting costfun and gradient
    u_prev,J_prev,∇J_prev = u,J,∇J
 #    print>0 && println(" | smooth($μ):
 # |       ls_options = $ls_options
 # |       ls_options_fallback = $ls_options_fallback")
    print>0 && println(" | smooth($μ):")
    for outer i = 1:μ
        d = i==1 ? -∇J : getdir(∇J,∇J_prev,d) # compute the new search direction
        s,_,_ = linesearch(u,J,∇J,d,f,ls_options)

        linesearch_fail = false
        s_bound = ( (:s_bound in propertynames(ls_options)) ? ls_options.s_bound(u,d) : Inf)::Float64
        if s < 0 || s > s_bound # Check the admissibility of s (and therefore the next iterate u)
            linesearch_fail = true
            print>0 && println(" |  ($i) J = $J, ‖∇J‖ = $(norm_∇J(∇J)), s=$s (s is not admissible)")
        else
            u_try = u + s*d
            J_try,∇J_try = f(u_try) # NOTE: For quadratic problems, one may obtain the gradient at the new point from the quadmin linesearch for free.
            if J_try >= J # Checks descent
                linesearch_fail = true
                print>0 && println(" |  ($i) J = $J, ‖∇J‖ = $(norm_∇J(∇J)), s=$s (no descent: next J would be $J_try)")
                # d = -∇J_prev
            end
        end

        if linesearch_fail && ls_options_fallback.method!="none"
            print>0 && println(" |       Trying a linesearch specified by ls_options_fallback.")
            # d = -∇J_prev
            s,_,_ = linesearch(u,J,∇J,d,f,ls_options_fallback)
            u_try = u + s*d
            J_try,∇J_try = f(u_try)
            if J_try >= J
                print>0 && println(" |       Still no descent: next J would be $J_try. Trying again with search direction -∇J.")
                d = -∇J_prev
                s,_,_ = linesearch(u,J,∇J,d,f,ls_options_fallback)
                u_try = u + s*d
                J_try,∇J_try = f(u_try)
                if J_try >= J
                    #throw(DebugException("No descent", (u=u,∇J=∇J_prev)))
                    print>0 && println(" |       WARNING! No descent. next J WILL be $(J_try)!")
                end
            end
        end

        print>0 && println(" |  ($i) J = $J, ‖∇J‖ = $(norm_∇J(∇J)), s=$s")

        # Finalize update
        u_prev,J_prev,∇J_prev = u,J,∇J
        u,J,∇J = u_try,J_try,∇J_try

        if breakcond(∇J); break; end
    end
    #DEBUG println(" | SMOOTHER DEBUG: J₀ = $J₀ and J = $J and J₀-J = $(J₀-J)")
    return u, J, ∇J, J₀, ∇J₀, i
end

"""
    mgoptV(u,v,f,I,k[,K])

Implementation of MG/OPT V-cycle.

# Arguments
- `u`: initial guess.
- `v`: corrective term.
- `f::Function`: function for calculating (J,∇J) given control u (discretized at level k) and MG/OPT level k.
- `I::Function`: level mapping function (u,k) ↦ u mapped to level k.
- `k::Int`: current MG/OPT level.
- `μ_pre::Vector{Int}`: μ_pre[k] gives the number of presmoothing steps at level k.
- `μ_post::Vector{Int}`: μ_post[k] gives the number of postsmoothing steps at level k.
- `K::Int`: finest MG/OPT level (defaults to k).

# Additional arguments
- `smooth::Function=smooth`: The smoother used. Should be callable as smooth(f,u₀,μ) with f(u)↦(J(u),∇J(u)), u₀ the starting value in the iteration and μ the number of smoothing steps.
- `inner::Function=dot`: The inner product w.r.t. which the gradient is defined.
- `print::Bool=false`: set to `true` to give some information
- `kmin::Int=0`: The depth that the V-cycle should reach.
- `ls_options`: how the step size is determined. A named tuple
    (method="constant",stepsize=1.0) (default): Recommended close to the solution
    (method="quadmin",): Recommended for linear problems far away from the solution.
    (method="armijo",inner,...): Recommended for nonlinear problems
        inner should be a function providing the inner product w.r.t. which the gradient is defined. For more options, see [`armijo`](@ref).

# Returns
- approximation of minimizer of f(u)[1] - v⋅u
- Costfunction at that approximation
- Gradient at that approximation
- Costfunction at the start of the V-cycle
- Gradient at the start of the V-cycle
"""
function mgoptV(u,v,f::Function,lm::Function,k::Int, μ_pre::Vector{Int} ,μ_post::Vector{Int}, K::Int=k; smooth::Function=(u,f,μ,k)->smooth(u,f,μ), inner::Function=dot, print::Int=0, kmin::Int=0, ls_options=(method="constant",stepsize=1.0))
    DEBUG = false

    norm(x) = sqrt(inner(x,x))
    ⋅ = inner

    # function modified by v to optimize
    function fv(u)
        J1,∇J1 = f(u,k)
        Jv = J1 - v⋅u
        ∇Jv = ∇J1 - v
        return Jv,∇Jv
    end

    print>0 && k==K && println("MG/OPT started. μ_pre = $μ_pre. μ_post = $μ_post.")

    if k<=kmin #trivial case
        # NOTE: One may also solve the optimization problem with 𝔼[k(x,ω)] as system parameters
        # or do a full Newton step since the Hessian will now be small anyway.
        DEBUG && println(" | debug: u=$u, v=$v")
        u,J,∇J,J₀,∇J₀,μdone=smooth(u,fv,μ_pre[k+1]+μ_post[k+1],k)
        print>0 && println(" | k = $k: After solve($μdone): ‖∇J‖ = $(norm(∇J))")
        return u,J,∇J,J₀,∇J₀
    end

    # pre-smoothing
    print>0 && println(" | k = $k: Presmoothing...")
    u,J,∇J,J₀,∇J₀,μdone=smooth(u,fv,μ_pre[k+1],k) #NOTE: J,∇J here contains -v part, since fv is supplied!!!
    print>0 && println(" | k = $k: After presmoothing($μdone): ‖∇J‖ = $(norm(∇J))")
    DEBUG && println(" | MG/OPT DEBUG: J₀=$J₀")

    # recursion
    uc = lm(u,k-1)
    _,∇Jc = f(uc,k-1) #NOTE: In theory free, but cost negligible compared to fine gradient
    J,∇J = f(u,k) #NOTE: should be obtained in smoothing step
    vc = lm(v,k-1) + ∇Jc - lm(∇J,k-1) #NOTE: This last correction can in theory come for free
    uc_new = mgoptV(uc,vc,f,lm,k-1,μ_pre,μ_post,K; smooth=smooth, inner=inner, print=print, kmin=kmin, ls_options=ls_options)[1]
    ec = uc_new - uc
    e = lm(ec,k)

    # linesearch
    s,_,_ = linesearch(u,J-v⋅u,∇J-v,e,fv,ls_options)
    print>1 && println(" | k = $k: step = $s")

    u = u + s*e

    # post-smoothing
    print>0 && println(" | k = $k: Postsmoothing...")
    # try
    #    u,J,∇J,J_debug,∇J_debug,μdone=smooth(u,fv,μ_post[k+1],k)
    #catch e
    #    throw(DebugException(e,"",(u=u,v=v)))
    #end
    u,J,∇J,J_debug,∇J_debug,μdone=smooth(u,fv,μ_post[k+1],k)
    print>0 && println(" | k = $k: After postsmoothing($μdone): ‖∇J‖ = $(norm(∇J))")
    DEBUG && println(" | MG/OPT DEBUG: J₀=$J₀, J_debug=$J_debug, J=$J, ")
    DEBUG && println(" | J₀-J_debug = $(J₀-J_debug) and J_debug-J = $(J_debug-J)")
    return u,J,∇J,J₀,∇J₀
end

"""
    mgoptW(u,v,f,I,k[,K])

Implementation of MG/OPT W-cycle.

# Arguments
- `u`: initial guess.
- `v`: corrective term.
- `f::Function`: function for calculating (J,∇J) given control u (discretized at level k) and MG/OPT level k.
- `I::Function`: level mapping function (u,k) ↦ u mapped to level k.
- `k::Int`: current MG/OPT level.
- `μ_pre::Vector{Int}`: μ_pre[k] gives the number of presmoothing steps at level k.
- `μ_post::Vector{Int}`: μ_post[k] gives the number of postsmoothing steps at level k.
- `K::Int`: finest MG/OPT level (defaults to k).

# Additional arguments
- `smooth::Function=smooth`: The smoother used. Should be callable as smooth(f,u₀,μ) with f(u)↦(J(u),∇J(u)), u₀ the starting value in the iteration and μ the number of smoothing steps.
- `inner::Function=dot`: The inner product w.r.t. which the gradient is defined.
- `print::Bool=false`: set to `true` to give some information
- `kmin::Int=0`: The depth that the W-cycle should reach.
- `ls_options`: how the step size is determined. A named tuple
    (method="constant",stepsize=1.0) (default): Recommended close to the solution
    (method="quadmin",): Recommended for linear problems far away from the solution.
    (method="armijo",inner,...): Recommended for nonlinear problems
        inner should be a function providing the inner product w.r.t. which the gradient is defined. For more options, see [`armijo`](@ref).

# Returns
- approximation of minimizer of f(u)[1] - v⋅u
- Costfunction at that approximation
- Gradient at that approximation
- Costfunction at the start of the W-cycle
- Gradient at the start of the W-cycle
"""
function mgoptW(u,v,f::Function,lm::Function,k::Int, μ_pre::Vector{Int} ,μ_post::Vector{Int}, K::Int=k; smooth::Function=(u,f,μ,k)->smooth(u,f,μ), inner::Function=dot, print::Int=0, kmin::Int=0, ls_options=(method="constant",stepsize=1.0))
    DEBUG = false

    norm(x) = sqrt(inner(x,x))
    ⋅ = inner

    # function modified by v to optimize
    function fv(u)
        J1,∇J1 = f(u,k)
        Jv = J1 - v⋅u
        ∇Jv = ∇J1 - v
        return Jv,∇Jv
    end

    print>0 && k==K && println("MG/OPT started. μ_pre = $μ_pre. μ_post = $μ_post.")

    if k<=kmin #trivial case
        # NOTE: One may also solve the optimization problem with 𝔼[k(x,ω)] as system parameters
        # or do a full Newton step since the Hessian will now be small anyway.
        DEBUG && println(" | debug: u=$u, v=$v")
        u,J,∇J,J₀,∇J₀,μdone=smooth(u,fv,2μ_pre[k+1]+2μ_post[k+1],k) #NOTE: factor 2 here
        print>0 && println(" | k = $k: After solve($μdone): ‖∇J‖ = $(norm(∇J))")
        return u,J,∇J,J₀,∇J₀
    end

    # pre-smoothing
    print>0 && println(" | k = $k: Presmoothing...")
    u,J,∇J,J₀,∇J₀,μdone=smooth(u,fv,μ_pre[k+1],k) #NOTE: J,∇J here contains -v part, since fv is supplied!!!
    print>0 && println(" | k = $k: After presmoothing($μdone): ‖∇J‖ = $(norm(∇J))")
    DEBUG && println(" | MG/OPT DEBUG: J₀=$J₀")

    # recursion
    uc = lm(u,k-1)
    _,∇Jc = f(uc,k-1) #NOTE: In theory free, but cost negligible compared to fine gradient
    J,∇J = f(u,k) #NOTE: should be obtained in smoothing step
    vc = lm(v,k-1) + ∇Jc - lm(∇J,k-1) #NOTE: This last correction can in theory come for free
    if k<=kmin+1 #W-cycle below will be trivial case
        uc_new = mgoptW(uc,vc,f,lm,k-1,μ_pre,μ_post,K; smooth=smooth, inner=inner, print=print, kmin=kmin, ls_options=ls_options)[1]
    else
        μ_pre1 = μ_pre[1:k]; μ_post1 = [μ_post[1:k-1];2*μ_post[k]]; #additional level k post smoothing here
        uc_new = mgoptW(uc,vc,f,lm,k-1,μ_pre1,μ_post1,K; smooth=smooth, inner=inner, print=print, kmin=kmin, ls_options=ls_options)[1]
        μ_pre2 = [μ_pre[1:k-1];0]; μ_post2 = μ_post[1:k]; #reduced level k pre smoothing here
        uc_new = mgoptW(uc_new,vc,f,lm,k-1,μ_pre2,μ_post2,K; smooth=smooth, inner=inner, print=print, kmin=kmin, ls_options=ls_options)[1]
    end
    ec = uc_new - uc
    e = lm(ec,k)

    # linesearch
    s,_,_ = linesearch(u,J-v⋅u,∇J-v,e,fv,ls_options)
    print>1 && println(" | k = $k: step = $s")

    u = u + s*e

    # post-smoothing
    print>0 && println(" | k = $k: Postsmoothing...")
    # try
    #    u,J,∇J,J_debug,∇J_debug,μdone=smooth(u,fv,μ_post[k+1],k)
    #catch e
    #    throw(DebugException(e,"",(u=u,v=v)))
    #end
    u,J,∇J,J_debug,∇J_debug,μdone=smooth(u,fv,μ_post[k+1],k)
    print>0 && println(" | k = $k: After postsmoothing($μdone): ‖∇J‖ = $(norm(∇J))")
    DEBUG && println(" | MG/OPT DEBUG: J₀=$J₀, J_debug=$J_debug, J=$J, ")
    DEBUG && println(" | J₀-J_debug = $(J₀-J_debug) and J_debug-J = $(J_debug-J)")
    return u,J,∇J,J₀,∇J₀
end

end



# # W-cycle test
# if k<=kmin+1
#     uc_new = mgoptW(uc,vc,f,lm,k-1,μ_pre[1:1],μ_post[1:1],K; smooth=smooth, inner=inner, print=print, kmin=kmin, ls_options=ls_options)[1]
# else
#     μ_pre1 = μ_pre[1:k]; μ_post1 = [μ_post[1:k-1];2*μ_post[k]];
#     uc_new = mgoptW(uc,vc,f,lm,k-1,μ_pre1,μ_post1,K; smooth=smooth, inner=inner, print=print, kmin=kmin, ls_options=ls_options)[1]
#     μ_pre2 = [μ_pre[1:k-1];0]; μ_post2 = μ_post[1:k];
#     uc_new = mgoptW(uc_new,vc,f,lm,k-1,μ_pre2,μ_post2,K; smooth=smooth, inner=inner, print=print, kmin=kmin, ls_options=ls_options)[1]
# end
