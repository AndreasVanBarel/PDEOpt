module Volume_C_Solver

using LinearAlgebra
using SparseArrays

export solve_state!, solve_adjoint!, solve_adjoint, cost, cost_grad_state

# average operator to obtain values of the diffusion coefficient in intermediate locations
⊙(x,y) = 2x*y/(x+y)

# Efficient sparse matrix discretization of the 2d Poisson PDE
function getA(k::Matrix{Float64}, Δx, Δy)
    nx,ny = size(k)
    m = (nx-2)*(ny-2) # A ∈ ℝᵐˣᵐ
    I = Int[]
    J = Int[]
    V = Float64[]
    spstore(i::Int,j::Int,v::Float64) = (push!(I,i); push!(J,j); push!(V,v))
    for y = 2:ny-1, x = 2:nx-1 #loop over all equations/internal points
        i = (x-1)+(y-2)*(nx-2) #; println("x = $x, y = $y, i = $i, ny = $ny") # debug
        y!=2 ? spstore(i,i-nx+2,-(k[x,y]⊙k[x,y-1])/Δy^2) : 0
        x!=2 ? spstore(i,i-1,-(k[x,y]⊙k[x-1,y])/Δx^2) : 0
        spstore(i,i,(k[x,y]⊙k[x-1,y] + k[x,y]⊙k[x+1,y])/Δx^2 + (k[x,y]⊙k[x,y-1] + k[x,y]⊙k[x,y+1])/Δy^2)
        x!=nx-1 ? spstore(i,i+1,-(k[x,y]⊙k[x+1,y])/Δx^2) : 0
        y!=ny-1 ? spstore(i,i+nx-2,-(k[x,y]⊙k[x,y+1])/Δy^2) : 0
    end
    A = sparse(I,J,V,m,m)
end

# PDE equation solver
# !y (nx × ny) is the state, assumed to already contain the boundary conditions at its boundary.
# u (nx × ny) is the rhs control. (note: boundary points are not used.)
# k (nx × ny) is the diffusion coefficient. (note: corner points are not used.)
function solve_state!(y::Matrix{T}, u::Matrix{T}, k::Matrix{T}, Δx::Real, Δy::Real) where T<:Real
    # determine sizes
    nx,ny = size(y)

    # generate A
    A = getA(k,Δx,Δy)

    # generate rhs
    rhs = u[2:nx-1,2:ny-1] # copy since changes will happen at the boundary
    rhs[:,1] .+= y[2:nx-1,1].*(k[2:nx-1,2].⊙k[2:nx-1,1])./Δy^2
    rhs[:,ny-2] .+= y[2:nx-1,ny].*(k[2:nx-1,ny-1].⊙k[2:nx-1,ny])./Δy^2
    rhs[1,:] .+= y[1,2:ny-1].*(k[2,2:ny-1].⊙k[1,2:ny-1])./Δx^2
    rhs[nx-2,:] .+= y[nx,2:ny-1].*(k[nx-1,2:ny-1].⊙k[nx,2:ny-1])./Δx^2
    rhs = vec(rhs)

    y[2:nx-1,2:ny-1] = A\rhs
    return y
end

# PDE adjoint equation solver
# !p (nx × ny) is the adjoint variable. p is an nx × ny Array; no other assumptions are made. It may be uninitialized.
# y (nx × ny) is the state.
# u (nx × ny) is the rhs control. (note: boundary points are not used.)
# z (nx × ny) is the target state.
# k (nx × ny) is the diffusion coefficient. (note: corner points are not used.)
function solve_adjoint!(p::Matrix{T}, y::Matrix{T}, u::Matrix{T}, z::Matrix{T}, k::Matrix{T}, Δx::Real, Δy::Real) where T<:Real
    # determine sizes
    nx,ny = size(y)

    # set the correct boundaries of p
    p[2:nx-1,1] .= 0
    p[1,:] .= 0
    p[nx,:] .= 0
    p[2:nx-1,ny] .= 0

    # generate A
    A = getA(k,Δx,Δy)

    # generate rhs
    nx,ny = size(y)
    rhs = view(y,2:nx-1,2:ny-1).-view(z,2:nx-1,2:ny-1)
    rhs = vec(rhs)

    p[2:nx-1,2:ny-1] = A\rhs
    return p
end

# Generates the adjoint p from solve_adjoint!(…) above.
solve_adjoint(y::Matrix{T}, u::Matrix{T}, z::Matrix{T}, k::Matrix{T}, Δx::Real, Δy::Real) where T<:Real = solve_adjoint!(Array{T}(undef,size(y)...),y,u,z,k,Δx,Δy)

function cost(u::Matrix{T}, z::Matrix{T}, k::Matrix{T}, Δx::Real, Δy::Real, α) where T<:Real
    nx,ny = size(k)
    y = Array{T}(undef,nx,ny)
    #setboundary!(y,bc)
    y[:,1].=0; y[1,:].=0; y[:,end].=0; y[end,:].=0; #set boundary condition
    solve_state!(y, u, k, Δx, Δy) # sets y
    J = 0.5*Δx*Δy*( sum((y-z).^2) + α*sum(u.^2) )
end

function cost_grad_state(u::Matrix{T}, z::Matrix{T}, k::Matrix{T}, Δx::Real, Δy::Real, α) where T<:Real
    nx,ny = size(k)
    y = Array{T}(undef,nx,ny)
    y[:,1].=0; y[1,:].=0; y[:,end].=0; y[end,:].=0; #set boundary condition
    solve_state!(y, u, k, Δx, Δy) # sets y
    J = 0.5*Δx*Δy*( sum((y-z).^2) + α*sum(u.^2) )
    p = solve_adjoint(y,u,z,k,Δx,Δy) # gets p
    ∇J = p .+ α.*u
    return J,∇J,y
end


end
