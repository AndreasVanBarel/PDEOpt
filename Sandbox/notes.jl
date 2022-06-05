#3/1/2020 - NaNs in Burger's MG/OPT
#
#exp_B_setup.jl + exp3.jl for problems3[6] generates NaNs.
#It appears that the generation of NaNs in the mgoptV for the Burgers' problem can be reproduced using the following data:
u0=[0.0, 0.394083, 0.50176, 0.362396, 0.209446, 0.105351, 0.0355882, 0.000252193, 0.0]
v=[0.0, 0.00421376, -0.0355065, -0.0302601, -0.0249538, -0.0191757, -0.0104197, -0.00235877, 0.0]
#Here, u is the current control, v the MG/OPT correction factor. The NaNs are produced when mgoptV calls "
#u,∇J,∇J_start=smooth(u,fv,μ_pre[k+1]+μ_post[k+1])
#with smooth, as defined in exp_B_setup.jl and fv of course equal to"
function fv(u)
    J₀,∇J₀ = f(u,0)
    Jv = J₀ - inner_∇J(v,u)
    ∇Jv = ∇J₀ - v
    return Jv,∇Jv
end
#with f being defined in exp3.jl"
#One may reproduce the bug (the generation of NaNs) by executing "
smoother_temp1(f,u,μ) = Optimization.smooth(f, u, μ; breakcond=g->norm_∇J(g)<1e-10, linesearch=(method="armijo",inner=inner_∇J))
u,∇J,∇J_start=smoother_temp1(u0,fv,16)
norm_∇J(∇J)
#Note that the result is much better if simple quadmin is used here:
smoother_temp2(f,u,μ) = Optimization.smooth(f, u, μ; breakcond=g->norm_∇J(g)<1e-10, linesearch=(method="quadmin",))
u,∇J,∇J_start=smoother_temp2(u0,fv,16)
norm_∇J(∇J)

## Problematic quantities in the Burgers' equation solver for Problem problems3[6]
u=[0.0, -0.0431987, 0.33783, 0.151604, 0.154609, 0.14817, 0.0455301, -0.0227936, 0.0]
v=[0.0, -0.00192711, 0.000542012, -0.00892142, -0.0099983, -0.00542617, -0.00290749, -0.0012653, 0.0]
g=[0.0, 9.63706e-6, 1.2119e-5, 3.32167e-5, 7.06684e-6, 3.14817e-5, 2.90831e-5, 4.08546e-6, 0.0]
d=[-0.0, -9.63706e-6, -1.2119e-5, -3.32167e-5, -7.06684e-6, -3.14817e-5, -2.90831e-5, -4.08546e-6, -0.0]
k = 1e-2*ones(size(u))
mesh = RegularGrid1D(length(u))
inner_∇J(a,b) = inner(a,b,mesh)

# We now construct the problematic costgrad function.
# u - The control input
# k - The random parameters
# v - The corrective term
function cost_grad(u::Vector{Float64}, k::Vector{Float64}, v::Vector{Float64}, inner::Function)
    ⋅ = inner
    m = 9; mt = 10000; T=1; s=-1.0; α=0.0;
    zfun = x->0.1(1-cos(2π*x))
    Δx = 1/(m-1)
    Δt = T/(mt-1)
    nodes = LinRange(0,1,m)
    z = zfun.(nodes)
    J₀,∇J₀,_ = Solver.cost_grad_state(u, z, k, ones(m).*s, mt, Δx, Δt, α)
    J₀ = 0*J₀; ∇J₀ = 0.0.*∇J₀;
    Jv = J₀ - v⋅u
    ∇Jv = ∇J₀ - v
    return Jv,∇Jv
end
# Test the evaluation of the function
J,∇J = cost_grad(u,k,v,inner_∇J)

# test gradient by comparing to finite differences result.
ϵ = 1e-8 #perturbation used in finite differences
∇J_diff1 = Array{Float64}(undef,size(u))
J = u->cost_grad(u,k,v,inner_∇J)[1]
for i = 1:length(u) #linear indexing
    δu = zeros(size(u)); δu[i] = ϵ
    ∇J_diff1[i] = (J(u + δu) - J(u))/ϵ
end
∇J_diff = ∇J_diff1.*(length(u)-1) #ensure correct scaling
pp.newfig(11); pp.plot(getproperty.(mesh,:x),∇J_diff)
pp.newfig(12); pp.plot(getproperty.(mesh,:x),∇J)
pp.newfig(13); pp.plot(getproperty.(mesh,:x),v)
e = ∇J .- ∇J_diff
pp.newfig(14); pp.plot(getproperty.(mesh,:x),e)


#This test checks the provided costfunction/gradient pair for consistency.
#f - function yielding (J,∇J) pair
#u - input value
#d - direction in which to compute. If empty, chooses gradient in u
function compute_line(f::Function, u, d)
    g = u->f(u)[2]
    J,∇J = f(u)
    s = quadmin(u,∇J,d,g)[1]
    steps = range(0, length = 20, stop = 2s)
    Js∇Js = [f(u+d*step) for step in steps]
    Js = [t[1] for t in Js∇Js]
    ∇Js = [t[2] for t in Js∇Js]
    return steps, Js, ∇Js
end
compute_line(f::Function, u) = compute_line(f,u,-f(u)[2])
