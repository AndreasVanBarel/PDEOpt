##################################
# Testing module Burgers_Solver #
##################################

## Initialize the test (run this block always)
import Burgers_Solver
using General
Solver = Burgers_Solver

## GENERAL TEST 2049,8193
nx = 2049 #33 65 2049
nt = 8193 #101 1001 8193 #minimum working in practice is 63
T = 1.0
k_value = 5e-4

Δx = 1/(nx-1)
Δt = T/(nt-1)

#stability condition
max_state = 1 #assumption
max_Δt = Δx^2/(max_state*Δx+2k_value)
min_nt = ceil(Int,1/max_Δt + 1)
min_nt < nt || error("Stability condition might not be satisfied!")

# variables
nodesx = LinRange(0,1,nx)
nodest = LinRange(0,T,nt)
mesh = RegularGrid2D(nodesx, nodest)
meshx = RegularGrid1D(nodesx)
y = zeros(nx,nt)
y_prime = zeros(nx,nt)
k = fill(k_value,nx)
#k[1:3] .= k_value*100; k[end-2:end] .= k_value*100
s = .-ones(nx)
blob(a,b) = x-> a<x<b ? 0.5-0.5*cos((x-a)*2π/(b-a)) : 0.0
ufun = x->blob(0.1,0.5)
#ufun = x->0.7<x<0.9 ? 1.0 : 0.0
#u = collect(0.0.*nodesx); u[ceil(Int,nx/2)]=1.0
u = ufun.(nodesx)
z = sin.(LinRange(0,π,nx))
#u.=z

@time Solver.solve_state!(y,u,k,s,Δx,Δt;y_prime=y_prime)
cm = pp.ColorMap("viridis")
pp.newfig(1); pp.surf(mesh,y); pp.xlabel("x"); pp.ylabel("t")
pp.plot(mesh,y,2); pp.xlabel("x"); pp.ylabel("t");
pp.plot(mesh,y_prime,3)
@time p = Solver.solve_adjoint(y,y_prime,u,z,k,s,Δx,Δt)
pp.newfig(4); pp.surf(p,cmap=cm,antialiased=false)
pp.plot(mesh,p,5)

# costfun and gradient
J = Solver.cost(u,z,k,s,nt,Δx,Δt,0.0)
J,∇J,y = Solver.cost_grad_state(u,z,k,s,nt,Δx,Δt,0.0)
pp.newfig(6); pp.plot(getproperty.(meshx,:x),∇J)

# test gradient by comparing to finite differences result.
ϵ = 1e-8 #perturbation used in finite differences
∇J_diff1 = Array{Float64}(undef,size(u))
J = u->Solver.cost(u,z,k,s,nt,Δx,Δt,0.0)
for i = 1:length(u) #linear indexing
    δu = zeros(size(u)); δu[i] = ϵ
    ∇J_diff1[i] = (J(u + δu) - J(u))/ϵ
end
∇J_diff = ∇J_diff1.*(length(u)-1) #ensure correct scaling
pp.newfig(7); pp.plot(getproperty.(meshx,:x),∇J_diff)
e = ∇J .- ∇J_diff
q = ∇J./∇J_diff
pp.newfig(8); pp.plot(getproperty.(meshx,:x),e)

# close up view of the gradient at a difficult point
pp.newfig(0); pp.surf(p[10:30,1:20],cmap=cm,antialiased=false)

## NOTE
# The following 33-long input generates instability for with a 1001 time step solve.
u = [0.0, 0.184777, 0.944089, 0.376754, 0.699407, 1.0588, 0.983237, 0.738166, 0.750602, 0.70945, 0.499953, 0.226997, -0.023534, -0.180602, -0.187614, -0.0617693, -0.0325907, 0.483558, 1.52863, 0.228073, 1.21168, 0.180275, 0.284765, 0.565257, 0.474108, 0.337838, 0.239891, 0.159594, 0.0933849, 0.0406324, 0.00552037, -0.00752104, 0.0]
pp.newfig(10); pp.plot(meshx,u);
# Set this u and run the above tests.

# At time step 800, we have
t_inspect = 100;
pp.newfig(11); pp.plot(meshx,y[:,t_inspect]);
pp.newfig(12); pp.surf(y[:,1:t_inspect],cmap=cm,antialiased=false)
# Clearly, the instability starts from the 0-boundary towards which the state flows. This suggests that something went wrong at that boundary.

## SECOND ORDER ACCURACY TEST
# The following code aims to numerically show the second order accuracy of the MacCormack method
blob(a,b) = x-> a<x<b ? 0.5-0.5*cos((x-a)*2π/(b-a)) : 0.0
T = 1.0
k_value = 5e-4
s_value = -1.0
ufun = blob(0.4,0.6)
zfun = blob(0.0,1.0)

function test_MacCormack(nx,nt,max_state=0.0)
    Δx = 1/(nx-1)
    Δt = T/(nt-1)

    # test stability condition
    max_Δt = Δx^2/(max_state*Δx+2k_value)
    min_nt = ceil(Int,1/max_Δt + 1)
    min_nt < nt || error("Stability condition might not be satisfied! nt must be larger than $min_nt but is currently $nt")

    nodesx = LinRange(0,1,nx)
    nodest = LinRange(0,T,nt)
    mesh = RegularGrid2D(nodesx, nodest)
    meshx = RegularGrid1D(nodesx)
    y = zeros(nx,nt)
    y_prime = zeros(nx,nt)
    k = k_value.*ones(nx)
    s = s_value.*ones(nx)
    u = ufun.(nodesx)
    z = zfun.(nodesx)

    Solver.solve_state!(y,u,k,s,Δx,Δt;y_prime=y_prime), mesh
end

y_ref, mesh_ref = test_MacCormack(2049,8193,1.0)
cm = pp.ColorMap("viridis")
pp.newfig(1); pp.surf(mesh_ref, y_ref)
pp.plot(mesh_ref,y_ref,2)

Es = []
nxs = []
nts = []
for i=1:10
    nx = 1+2^i; push!(nxs,nx)
    nt = 1+4*2^10; push!(nts,nt) # nt = 1+4*2^i; push!(nts,nt)
    y,_ = test_MacCormack(nx,nt)
    q = 2^(11-i)
    e = y-y_ref[1:q:end,1:2:end] # e = y-y_ref[1:q:end,1:q:end]
    push!(Es,sqrt(sum(e.^2)/(nx*nt)))
end

pp.newfig(3)
pp.plot(log2.(nxs.*nts),log2.(Es))
pp.plot(log2.(nxs.*nts),-2.0.*log2.(nxs.*nts).+2.0*log2(nxs[1]*nts[1])) #reference line
