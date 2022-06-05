module Stoch
# Module for sampling the stochastic field
# The most important function:
# - gen_sampler(seed::Int, covfun::HomogeneousCovFun, mesh::Mesh)
#   yields Function sampler(i::Int)

using General # Points, Mesh, Hierarchies
using FFTW
using Random
using LinearAlgebra

import Base: eltype, ndims, size, length, axes, getindex, show, +, -, *, /, \
import Base: Matrix
import LinearAlgebra: eigen, eigvals, eigvecs, isposdef

export Distribution, Gaussian, LogNormal, Uniform, Deterministic
export CovFun, HomogeneousCovFun, GeneralCovFun
export exponentialcovariance, materncovariance, rationalquadraticcovariance
export sample, gen_sampler
export ANestedCirculant, NestedCirculant, NestedSymmetricCirculant
export covariancematrix, circulantembed, pad, symmetrize, getblockstructure

###########################
### COVARIANCE FUNCTION ###
###########################
# CovFun takes two Points and calculates the covariance of the stochastic field between those points
abstract type CovFun end

struct GeneralCovFun <: CovFun
    fun::Function
end

struct HomogeneousCovFun <: CovFun #Homogeneous : only dependent on x-y
    fun::Function
end
show(io::IO, h::HomogeneousCovFun) = print(io, "HomogeneousCovFun (see ~.fun)")
show(io::IO, h::GeneralCovFun) = print(io, "GeneralCovFun (see ~.fun)")
(K::GeneralCovFun)(p1,p2) = K.fun(p1,p2)::Real
(K::HomogeneousCovFun)(delta) = K.fun(delta)::Real
(K::HomogeneousCovFun)(x,y) = K(x-y)::Real
exponentialcovariance(λ::Real, σ::Real, p::Real=1) = HomogeneousCovFun(δ->σ^2*exp(-norm(δ,p)/λ))
rationalquadraticcovariance(α::Real, l::Real, σ::Real, p::Real=1) = HomogeneousCovFun(δ->σ^2*(1+(δ/l)^2/(2α))^-α)
function materncovariance(v::Real, ρ::Real, σ::Real, p::Real=1) #v is a smoothness parameter, ρ is like λ, a correlation length metric
    isinf(v) && ( return HomogeneousCovFun(δ->σ^2*exp(-(δ/l)^2/2)) )
    Γ = gamma; Kᵥ = x->besselk(v,x);
    function materncovfun(δ)
        d = norm(δ,p)
        if d>1e-8 # normal evaluation
            return σ^2*2.0^(1-v)/Γ(v)*(√(2v)d/ρ)^v*Kᵥ(√(2v)d/ρ)
        else
            return σ^2
            #return σ^2*(1 + v/(2-2v)*(d/ρ)^2 + v^2/(8(2-3v+v^2))*(d/ρ)^4)  # fourth order Taylor expansion
        end
    end
    HomogeneousCovFun(materncovfun)
end

# NOTE: The result produced by covariancematrix is not guaranteed to be positive semidefinite!
covariancematrix(covfun::CovFun, mesh::Mesh) = reshape([covfun(p1,p2) for p1 in mesh, p2 in mesh],length(mesh),length(mesh))
covariancematrix(covfun::CovFun, points::Array) = reshape([covfun(p1,p2) for p1 in points, p2 in points],length(points),length(points))

#####################
### DISTRIBUTIONS ###
#####################
abstract type Distribution end
struct Gaussian <: Distribution
    mean::Function
    cov::CovFun
end
Gaussian(c::CovFun) = Gaussian(x->0,c)
struct LogNormal <: Distribution
    mean::Function
    cov::CovFun
end
LogNormal(c::CovFun) = LogNormal(x->0,c)
struct Uniform <: Distribution
    lowest::Float64
    highest::Float64
end
struct Deterministic <: Distribution
    fun::Function
end
Deterministic(value::Number) = Deterministic(x->value)

show(io::IO, d::Gaussian) = print(io, "Gaussian Distribution")
show(io::IO, d::LogNormal) = print(io, "LogNormal Distribution")
show(io::IO, d::Uniform) = print(io, "Uniform Distribution on [$(d.lowest),$(d.highest)]")
show(io::IO, d::Deterministic) = print(io, "Deterministic")

###########################
### CIRCULANT EMBEDDING ###
###########################
# Attempts to generate symmetric positive semidefinite nested circulant matrix
# in which the covariance matrix corresponding to the given HomogeneousCovFun and Mesh
function circulantembed(covfun::HomogeneousCovFun, grid::RegularGrid)
    topleft = grid[1] #first element of mesh
    extended_grid = grid; # start out with no extension
    padding = 0
    while padding < 4*sqrt(length(grid))
        A = reshape([covfun(y,topleft) for y in extended_grid],size(extended_grid))
        C = NestedSymmetricCirculant(A)
        isposdef(C) && (return C)
        extended_grid = extend(grid,padding+=ceil(Int,sqrt(length(grid))/4))
        #NOTE: padding only in one dimension worsens the spectrum!
    end
    error("Number of padding elements required to produce a positive semidefinite nested circulant matrix is unacceptably large. \n
    Is the given covariance function positive definite?")
end

# AUX: Fast implementation of the real to real discrete cosine transform of type 1
# Note: for real symmetric data v = [v₀, v₁, …, vₙ, vₙ₋₁, …, v₁], we have
# y = fft(v) = [y₀, y₁, …, yₙ, yₙ₋₁, …, y₁] is real and symmetric.
# This real arithmetic function fct can be used to speed up this computation since
# fct([v₀, v₁, …, vₙ]) = [y₀, y₁, …, yₙ].
# Call symmetrize(fct(v)) to reproduce fft(v) if needed.
fct(v::Array{<:Real},dims) = FFTW.r2r(v,FFTW.REDFT00,dims)
fct(v::Array{<:Real}) = FFTW.r2r(v,FFTW.REDFT00)

###############
### UTILITY ###
###############
# does padding for the matrix with zeros. The size of A is increased by n
function pad(A::AbstractArray{T,N}, n::Vararg{Int,N}) where {T,N}
    newsize = size(A).+n
    B = zeros(T,newsize...)
    B[axes(A)...] = A;
    return B
end
pad(A::AbstractArray{T,N}, n::Int) where {T,N} = pad(A, fill(n,N)...)

# Symmetrizes A in all dimensions.
# E.g., symmetrize([a₁,…,aₙ]) yields [a₁,…,aₙ,aₙ₋₁,…,a₂]
# Each dimension size nᵢ will become 2nᵢ-2
function symmetrize(A::AbstractArray{T,N}) where {T,N}
    dims = 2collect(size(A)).-2
    B = zeros(T,dims...)
    # iterate over all corners
    for corner = 0:2^N-1
        bs = Bool.(digits(corner,base=2,pad=N)) # bitrepresentation of corner number
        # build ranges for that corner
        rangesA = collect(bs[i] ? (2:size(A)[i]-1) : (1:size(A)[i]) for i in 1:N)
        rangesB = collect(bs[i] ? (size(B)[i]:-1:size(A)[i]+1) : (1:size(A)[i]) for i in 1:N)
        B[rangesB...] = A[rangesA...]
    end
    return B
end

#########################
### SAMPLER GENERATOR ###
#########################
hf(s::Int) = ( Random.seed!(s); abs(rand(Int)) ) # simple naive hashing function

# generates a sampler function for a given seed, distribution and mesh.
# The generated function has pre-computed eigenvalues (through fft)
function gen_sampler(seed::Int, d::Gaussian, grid::RegularGrid)
    C = circulantembed(d.cov,grid)
    λs = symmetrize(fct(C.A)) #calculate eigenvalues
    c = 1/√size(C,1)
    sqrtλs=c.*sqrt.(λs)
    transform = plan_rfft(λs; flags=FFTW.MEASURE)
    seed_derived = hf(seed)
    mean = apply(d.mean,grid)
    function sampler(i::Int) # function generates sample on the mesh it was constructed at.
        Random.seed!(seed_derived+i) # initializes random number generator with seed equal to seed_derived+i
        mean.+sample_(sqrtλs,grid; F=transform)
    end
    return sampler
end

function gen_sampler(seed::Int, d::LogNormal, mesh::Mesh)
    gaussian_sampler = gen_sampler(seed,Gaussian(d.mean,d.cov),mesh)
    return i::Int->exp.(gaussian_sampler(i))
end

function gen_sampler(seed::Int, d::Uniform, mesh::Mesh)
    seed_derived = hf(seed)
    function sampler(i::Int)
        Random.seed!(seed_derived+i)
        value = covfun.lowest + rand()*(covfun.highest-covfun.lowest)
        fill(value,size(mesh))
    end
end

function gen_sampler(seed::Int, d::Deterministic, mesh::Mesh)
    sampler(i::Int) = d.fun.(mesh)
end

function gen_sampler(seed::Int, d::Distribution, h::Hierarchy)
    samplers = [gen_sampler(seed, d, h.meshes[ℓ]) for ℓ in 1:length(h.meshes)]
    sampler(ℓ::Int, i::Int) = samplers[ℓ+1](i)
    return sampler
end

# internal function for efficient sampling given sqrtλs
# assuming:
#   all(size(sqrtλs).==size(points))
#   all(sqrtλs.>=0.0)
function sample_(sqrtλs::Array{<:Real}, grid::RegularGrid, points::Array{<:Real}=randn(size(sqrtλs)); F=plan_rfft(sqrtλs))
    # F contains the fft matrix (implicitly)
    complexdata = F*(sqrtλs.*points)
    data = real.(complexdata).+imag.(complexdata) # Discrete Hartley transform
    data[axes(grid)...] # cut out the relevant part
end

# NOTE: implementing the ordering of most important to least important point comparing sums,
# and then, to obtain full ordering, sort lexicographically.

# NOTE: the interface in MLMC for these functions would be to pass a function that can generate new independent gen_sample(ℓ,i) functions

########################
### NESTED CIRCULANT ###
########################
# A(bstract/ny)NestedCirculant
abstract type ANestedCirculant{E<:Number} <: AbstractMatrix{E} end
# tensor containing data. Each column (variable first index) corresponds to a
# circulant matrix. Each matrix (variable first and second index) corresponds
# to a block circulant matrix with each block circulant determined by the columns
# This easily generalizes to an arbitrary number of dimensions.
struct NestedCirculant{E<:Number} <: ANestedCirculant{E}
    A::AbstractArray{E}
end
# NestedSymmetricCirculant(A) representes NestedCirculant{symmetrize(A)}
struct NestedSymmetricCirculant{E<:Number} <: ANestedCirculant{E}
    # tensor containing data. Each column (variable first index) corresponds to a
    # circulant matrix. Each matrix (variable first and second index) corresponds
    # to a block circulant matrix with each block circulant determined by the columns
    # This easily generalizes to an arbitrary number of dimensions.
    A::AbstractArray{E}
end

# matrix functions
eltype(C::ANestedCirculant{E}) where E<:Number = E
ndims(C::ANestedCirculant) = 2;
size(C::NestedCirculant, n::Int) = length(C.A)
size(C::NestedSymmetricCirculant, n::Int) = prod(2 .*size(C.A).-2)
size(C::ANestedCirculant) = (size(C,1),size(C,2))
length(C::ANestedCirculant) = prod(size(C))
axes(C::ANestedCirculant, n::Int) = Base.OneTo(size(C,n))
axes(C::ANestedCirculant) = (axes(C,1),axes(C,2))
# AUX
# returns (s₁, …, sₙ) where n is equal to the amount of times the blocks are nested.
# for an unstructured matrix, n=1, for a matrix consisting of blocks of elements, n=2, etc.
# The input matrix then contains sₙ blocks in each dimension,
# each of which contain sₙ₋₁ blocks in each dimension,
# ...
# each of which contain s₁ elements in each dimension
# The basic (inner) blocks thus contain s₁×s₁ elements.
getblockstructure(C::NestedCirculant) = size(C.A)
getblockstructure(C::NestedSymmetricCirculant) = 2 .*size(C.A).-2
# AUX
#returns (block number,remainder index)
#assumes the matrix is partitioned in m blocks of size p×p
function getblocknumber(i,j,m,p)
    q1,r1 = divrem(i-1,p)
    q2,r2 = divrem(j-1,p)
    mod(q1-q2,m)+1,r1+1,r2+1
end
function getindex(C::NestedCirculant,i::Int,j::Int)
    s = size(C.A)
    p = prod(s)
    loc = Vector{Int}(undef,ndims(C.A))
    for d in ndims(C.A):-1:1
        p /= s[d]
        loc[d],i,j = getblocknumber(i,j,s[d],p)
    end
    C.A[loc...]
end
function getindex(C::NestedSymmetricCirculant,i::Int,j::Int)
    s = 2 .*size(C.A).-2 #s contains size of symmetrize(C.A)
    p = prod(s)
    loc = Vector{Int}(undef,ndims(C.A))
    for d in ndims(C.A):-1:1
        p /= s[d]
        loc[d],i,j = getblocknumber(i,j,s[d],p)
        if loc[d] > size(C.A,d); loc[d] = s[d]-loc[d]+2 end
    end
    C.A[loc...]
end
#TODO fix setindex to actually change all corresponding elements (low priority; setting never needed)
#setindex!(C::Circulant,i::Int,j::Int) = @error("setting index for a circulant matrix is not supported.")
#TODO make method more efficient (low priority; full matrix never needed)
Matrix(C::ANestedCirculant) = [C[i,j] for i in 1:size(C,1), j in 1:size(C,2)]

# arithmetic NestedCirculant
+(X::NestedCirculant, Y::NestedCirculant) = NestedCirculant(X.A+Y.A)
-(X::NestedCirculant, Y::NestedCirculant) = NestedCirculant(X.A-Y.A)
-(X::NestedCirculant) = NestedCirculant(-X.A)
*(X::NestedCirculant, c::Number) = NestedCirculant(X.A*c)
*(c::Number, X::NestedCirculant) = *(X,c) #only right vector multiplication implemented
function *(X::NestedCirculant, V::Vector{<:Number})
    length(V) == size(X,2) || (error("dimension mismatch"); return)
    s = size(V)
    T = fft(X.A).*fft(reshape(V,size(X.A)))
    reshape(real(ifft!(T)),s)
end

# arithmetic NestedSymmetricCirculant
+(X::NestedSymmetricCirculant, Y::NestedSymmetricCirculant) = NestedSymmetricCirculant(X.A+Y.A)
-(X::NestedSymmetricCirculant, Y::NestedSymmetricCirculant) = NestedSymmetricCirculant(X.A-Y.A)
-(X::NestedSymmetricCirculant) = NestedSymmetricCirculant(-X.A)
*(X::NestedSymmetricCirculant, c::Number) = NestedSymmetricCirculant(X.A*c)
*(c::Number, X::NestedSymmetricCirculant) = *(X,c)
function *(X::NestedSymmetricCirculant, V::Vector{<:Number}) #only right vector multiplication implemented
    length(V) == size(X,2) || (error("dimension mismatch"); return)
    s = size(V)
    T = symmetrize(fct(X.A)).*fft(reshape(V,2 .*size(X.A).-2))
    reshape(real(ifft!(T)),s)
end

# spectrum
eigen(X::ANestedCirculant) = Eigen(eigvals(X),eigvecs(X))
isposdef(X::ANestedCirculant) = all(eigvals(X).>0)
#TODO: For NestedSymmetricCirculant, the eigenvectors can all be made real
# (by linear combinations, e.g., summing/taking real and imag parts since they are conjugated anyway)
function eigvecs(X::ANestedCirculant{E}) where E
    es = Matrix{Complex{Float64}}(undef,size(X))
    for i = 1:size(X,1)
        m = zeros(E, getblockstructure(X));
        m[i] = 1
        es[:,i] = reshape(ifft(m),size(X,1))
    end
    return es
end
eigvals(X::NestedCirculant) = (fft(X.A))[:]
eigvals(X::NestedSymmetricCirculant) = (symmetrize(fct(X.A)))[:]

###############
### SAMPLER ###
###############
# function sample(d::Uniform, mesh::Mesh)
#     Random.seed!(seed_derived)
#     value = covfun.lowest + rand()*(covfun.highest-covfun.lowest)
#     fill(value,size(mesh))
# end

sample(d::Distribution, mesh::Mesh) = @error("Sampling of Distribution $d on Mesh $d not implemented") # fallback method
function sample(d::Gaussian, mesh::Mesh)
    g = sample_gaussian(d.cov,mesh)
    g .+= apply(d.mean,mesh)
end
sample(d::LogNormal, mesh::Mesh) = exp.(sample(Gaussian(d.mean, d.cov), mesh))

function sample_gaussian(covfun::CovFun, mesh::Mesh) # any mesh
    covmat = covariancematrix(covfun,mesh)
    noise=randn(length(mesh))
    E = eigen(Symmetric(covmat))
    λs,Q = E.values, E.vectors
    all(λs.>= 0.0) || error("given covariance function not positive semidefinite, min(λ) = $(minimum(λs))")
    reshape(Q*(sqrt.(λs).*noise),size(mesh))
end

function sample_gaussian(covfun::HomogeneousCovFun, grid::RegularGrid)
    covmat = circulantembed(covfun,grid)
    noise = randn(getblockstructure(covmat))
    λs = symmetrize(fct(covmat.A)) #calculate eigenvalues
    c = 1/√size(covmat,1)
    sample_(c.*sqrt.(λs), grid, noise)
end

end
