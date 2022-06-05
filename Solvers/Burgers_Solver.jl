module Burgers_Solver

using LinearAlgebra
using SparseArrays

export solve_state!, solve_adjoint!, solve_adjoint, cost, cost_grad_state

function shortstring(u::Vector)
    if length(u) > 17
        return string(u[1:4])[1:end-1]*", ... , "*string(u[end-3:end])[2:end]*"(length(u)=$(length(u)), max|u|=$(maximum(abs.(u))))"
    else
        return string(u)
    end
end

# PDE equation solver
# !y (nx × nt) - the state
# u (nx × 1) - the initial condition control. (note 1)
# k (nx × 1) - the diffusion coefficient (constant in time). (note 1)
# s (nx × 1) - the advection coefficient (constant in time). (note 1)
# (note 1: Boundary points are not used.)
function solve_state!(y::Matrix{T}, u::Vector{T}, k::Vector{T}, s::Vector{T}, Δx::Real, Δt::Real; y_prime::Matrix{T} = Matrix{T}(undef,size(y)), WARN_STAB = false) where T<:Real

    # check stability condition. NOTE: Putting these declarations in an if WARN_STAB is very slow!
    smallestmaxΔt = Inf
    tsmallestmaxΔt = 0.0
    stabproblem = false
    tstabproblem = Inf
    maxk = maximum(k)
    getmaxΔt(y_t) = Δx^2/(maximum(abs.(y_t))*Δx+2maxk)
    function check_stability(y_t,t)
        maxΔt = getmaxΔt(y_t)
        if maxΔt ≤ smallestmaxΔt
            smallestmaxΔt = maxΔt
            tsmallestmaxΔt = t
        end
        if maxΔt ≤ Δt
            #potential instability!
            tstabproblem = min(t,tstabproblem)
            stabproblem = true;
        end
    end

    # determine sizes
    nx,nt = size(y)

    # set initial and boundary conditions
    y[:,1] .= u
    y[1,:] .= 0
    y[nx,:] .= 0

    # for all timesteps:
    ytemp = Vector{T}(undef,nx)
    #ytemp[1] = 0; ytemp[end] = 0
    ψ(i,j) = 0.5*s[i]*y[i,j]^2
    ψ(i) = 0.5*s[i]*ytemp[i]^2
    for t = 1:nt-1
        WARN_STAB && check_stability(y[2:nx-1,t],t)
        ytemp[1] = (y[1,t]
                + (Δt/Δx) * (ψ(2,t)-ψ(1,t))
                + (Δt/Δx^2) * k[1] * (y[2,t]-2.0*y[1,t])
        )
        @views ytemp[2:nx-1] .= (y[2:nx-1,t]
                .+ (Δt/Δx) .* (ψ.(3:nx,t).-ψ.(2:nx-1,t))
                .+ (Δt/Δx^2) .* k[2:nx-1] .* (y[3:nx,t].-2.0.*y[2:nx-1,t].+y[1:nx-2,t])
        )
        ytemp[nx] = (y[nx,t]
                + (Δt/Δx) * (-ψ(nx,t))
                + (Δt/Δx^2) * k[nx] * (-2.0*y[nx,t]+y[nx-1,t])
        )
        # NOTE: for some reason this way of writing down the code results in less allocations, yet is slower overall. (More gc time supposedly)
        # @views y[2:nx-1,t+1] .= y[2:nx-1,t] .+ ytemp[2:nx-1] .+ (Δt/Δx) .* (ψ.(2:nx-1).-ψ.(1:nx-2))
        # @views y[2:nx-1,t+1] .= 0.5.*(y[2:nx-1,t+1] .+ (Δt/Δx^2) .* k[2:nx-1] .* (ytemp[3:nx].-2.0*ytemp[2:nx-1].+ytemp[1:nx-2]))
        @views y[2:nx-1,t+1] .= 0.5.*(y[2:nx-1,t] .+ ytemp[2:nx-1]
                .+ (Δt/Δx) .* (ψ.(2:nx-1).-ψ.(1:nx-2))
                .+ (Δt/Δx^2) .* k[2:nx-1] .* (ytemp[3:nx].-2.0*ytemp[2:nx-1].+ytemp[1:nx-2])
        )
        y_prime[:,t] .= ytemp
    end
    y_prime[:,end] .= 0
    WARN_STAB && check_stability(y[2:nx-1,nt],nt)

    if WARN_STAB && stabproblem
        @warn("Stability condition of the MacCormack method is no longer satisfied at time $((tstabproblem-1)*Δt) for u="*shortstring(u)*" ! Δt = $Δt, but at time $((tsmallestmaxΔt-1)*Δt), it must be smaller than $smallestmaxΔt.")
    end

    return y
end

# PDE adjoint equation solver
# !p (nx × ny) - the adjoint variable. p is an nx × ny Array; no other assumptions are made. It may be uninitialized.
# y (nx × ny) - the state.
# u (nx × 1) - the initial condition control. (note 1)
# z (nx × 1) - the target end state. (note 1)
# k (nx × 1) - the diffusion coefficient (constant in time). (note 1)
# s (nx × 1) - the advection coefficient (constant in time). (note 1)
# (note 1: Boundary points are not used.)
function solve_adjoint!(p::Matrix{T}, y::Matrix{T}, y_prime::Matrix{T}, u::Vector{T}, z::Vector{T}, k::Vector{T}, s::Vector{T}, Δx::Real, Δt::Real) where T<:Real
    # # amend k at the boundaries
    # k = copy(k)
    # k[2] = 0.1*k[2]
    # k[end-1] = 0.1*k[end-1]

    # determine sizes
    nx,nt = size(y)

    # set end and boundary conditions
    @views p[:,nt] .= z.-y[:,nt]
    p[1,:] .= 0
    p[nx,:] .= 0

    # for all timesteps:
    ptemp = Vector{T}(undef,nx)
    ptemp[1] = 0; ptemp[end] = 0
    for t = nt:-1:2
        @views ptemp[2:nx-1] .= (p[2:nx-1,t]
                .- s[2:nx-1].*y_prime[2:nx-1,t-1] .* (Δt/Δx).*(p[3:nx,t].-p[2:nx-1,t])
                .+ (k[3:nx].*p[3:nx,t].-2.0.*k[2:nx-1].*p[2:nx-1,t].+k[1:nx-2].*p[1:nx-2,t]).*(Δt/Δx^2) )
        @views p[2:nx-1,t-1] .= 0.5.*(p[2:nx-1,t] .+ ptemp[2:nx-1]
                .- s[2:nx-1].*y[2:nx-1,t-1] .* (Δt/Δx).*(ptemp[2:nx-1].-ptemp[1:nx-2])
                .+ (k[3:nx].*ptemp[3:nx].-2.0.*k[2:nx-1].*ptemp[2:nx-1].+k[1:nx-2].*ptemp[1:nx-2]).*(Δt/Δx^2)
        )
    end

    return p
end

# Generates the adjoint p from solve_adjoint!(…) above.
solve_adjoint(y::Matrix{T}, y_prime::Matrix{T}, u::Vector{T}, z::Vector{T}, k::Vector{T}, s::Vector{T}, Δx::Real, Δt::Real) where T<:Real = solve_adjoint!(Array{T}(undef,size(y)...),y,y_prime,u,z,k,s,Δx,Δt)

function cost(u::Vector{T}, z::Vector{T}, k::Vector{T}, s::Vector{T}, nt::Int, Δx::Real, Δt::Real, α::Real; WARN_STAB=false) where T<:Real
    nx = size(k,1)
    y = Array{T}(undef,nx,nt)
    solve_state!(y,u,k,s,Δx,Δt;WARN_STAB=WARN_STAB) # sets y
    J = 0.5/(nx-1)*(sum((y[:,end]-z).^2) + α*sum(u.^2))
end

# DEBUG: global lastcall
function cost_grad_state(u::Vector{T}, z::Vector{T}, k::Vector{T}, s::Vector{T}, nt::Int, Δx::Real, Δt::Real, α::Real; WARN_STAB=false) where T<:Real
    # DEBUG: global lastcall = (u, z, k, s, nt, Δx, Δt, α, WARN_STAB)
    nx = size(k,1)
    y = Array{T}(undef,nx,nt)
    y_prime = Array{T}(undef,nx,nt)
    solve_state!(y,u,k,s,Δx,Δt;y_prime=y_prime,WARN_STAB=WARN_STAB) # sets y and yprime
    J = 0.5/(nx-1)*(sum((y[:,end]-z).^2) + α*sum(u.^2))
    p = solve_adjoint(y,y_prime,u,z,k,s,Δx,Δt) # gets p
    ∇J = -p[:,1] .+ α.*u
    all(.!isnan.(y)) || error("Solve generated NaNs in y for u="*shortstring(u))
    !isnan(J) || error("Solve generated NaN for J for u="*shortstring(u))
    all(.!isnan.(∇J)) || error("Solve generated NaNs in ∇J for u="*shortstring(u))
    return J,∇J,y
end

end
