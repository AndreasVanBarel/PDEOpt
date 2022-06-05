#############################
# Testing module Stoch      #
#############################
using General #for definition of the meshes
using Stoch
using LinearAlgebra

#####  NestedCirculant and NestedSymmetricCirculant #####

## NestedCirculant
A = [100i+10j+k for i in 1:4, j in 1:3, k in 1:2]
A# = [10j+k for j in 1:4, k in 1:3]
C = NestedCirculant(A)

# arithmetic: multiplication
v = rand(size(C,2))
prod0 = C*v
norm(prod0 - Matrix(C)*v) #should be close to zero

# spectrum
# eigenvalues
islesscomplex(x::Complex,y::Complex) = abs(x)!=abs(y) ? abs(x)<abs(y) : real(x)<real(y)
λs = eigvals(C); λsorted = sort(λs; lt=islesscomplex, rev=true)
λs2 = eigvals(Matrix(C)); λ2sorted = sort(λs2; lt=islesscomplex, rev=true)
norm(λ2sorted - λsorted) #should be close to zero

# eigenvectors
es = (eigvecs(C))
norm((Matrix(C)*es./es)[1,:]-eigvals(C)) #should be close to zero
res = Matrix(C)*es - es*diagm(0=>eigvals(C))
norm(res) #should be close to zero

## Padding and symmetrizing
symA = symmetrize(A)
NC = NestedCirculant(symA)
norm(NC-NC') #should be exactly zero
NSC = NestedSymmetricCirculant(A)
norm(NSC-NC) #should be exactly zero

## NestedSymmetricCirculant

# arithmetic: multiplication
v = ones(size(NSC,2))
prod1 = NSC*v
norm(prod1 - Matrix(NSC)*v) #should be close to zero

# spectrum
# eigenvalues
λs = eigvals(NSC) # type should be Float64, not Complex
λsorted = sort(real(λs); rev=true)
λs2 = eigvals(Matrix(NSC)); λ2sorted = sort(λs2; rev=true)
norm(λ2sorted - λsorted) #should be close to zero

# eigenvectors
es = (eigvecs(NSC))
norm((Matrix(NSC)*es./es)[1,:]-eigvals(NSC)) #should be close to zero
res = Matrix(NSC)*es - es*diagm(0=>eigvals(NSC))
norm(res) #should be close to zero

##### Stochastic Field Generation #####
λ = 0.3
σ = 0.5
p = 2
covfun = exponentialcovariance(λ,σ,p)
g = Gaussian(x->0,covfun)

grid = RegularGrid2D(9,9)
circulantembed(covfun,grid)

grid = RegularGrid2D(257,257)
sam = sample(g,grid)
sampler = gen_sampler(0,g,grid)
pp.plot(grid, sampler(2),1)

# checking mean
function getμ(N::Int)
    sum = sampler(0)
    for i=1:N
        sum+=sampler(i)
    end
    return sum/(N+1)
end

@time μ = getμ(1000)
pp.surf(grid,μ,2); pp.title("μ = E[y]")

# recovering the covariance function
# Cov = 𝔼[(x-μ)(x-μ)ᵀ]
d = (sampler(0).-μ)
function getcov(N::Int)
    sum = d*d[1]'
    for i=1:N
        d .= sampler(i).-μ
        sum+=d*d[1]'
    end
    c = sum/(N+1)
end
@time c = getcov(1000)
c1 = reshape(c,size(grid))
exact_c1 = reshape([covfun(p,grid[1]) for p in grid],size(grid))
pp.surf(grid,c1,3); pp.title("first column of E[(x-μ)(x-μ)ᵀ]")
pp.surf(grid,c1.-exact_c1,3); pp.title("first column of Cov[y] - E[(x-μ)(x-μ)ᵀ]")

## NOTE: fct is its own inverse, up to a scaling constant of prod(2.*size(X).-2)
R = rand(5,5)
c = prod(2.0.*size(R).-2)
R-(Stoch.fct(Stoch.fct(R)))/c
