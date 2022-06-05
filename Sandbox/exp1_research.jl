## EXPERIMENT 1 ##
# Basic NCG fixed sample experiment #
# This gives an indication of the convergence speed of the optimization algorithm itself.

using Gradient
using Optimization

"ASSUME:
 c       - A ComputeStruct (see Gradient.~)
 u0      - A compatible starting value for the iteration."

# Experiment parameters
maxit = 12
#n = [1000,200,40,8]
#n = ceil.(Int,(1/1)*[5551, 949, 158, 32, 6, 1][end-prob.L:end])
#n = [250,125,63,32,16,8]
#n = [63,32,16,8]
n = fill(1,prob.L+1)

# Experiment
s = SamplingData(0,n)
its = []
function plotfun(k,u,J,∇J,𝔼y,𝕍y,d)
    k<10 && return
    mesh = h.meshes[end]
    mesh_u = h_sampler.meshes[end]
    pp.plot(mesh_u,u,100+k); pp.title("u")
    pp.plot(mesh_u,∇J,200+k); pp.title("∇J₀")
    pp.surf(mesh,𝔼y,300+k); pp.title("𝔼y")
    pp.surf(mesh,𝕍y,400+k); pp.title("𝕍y")
    pp.plot(mesh_u,d,500+k); pp.title("d")
end
@time u = ncg(u0, c, 1e-10, maxit, s; plotfun=plotfun, print=2, save=its, linesearch="sufficient_descent")

stepsizes = [its[i][8] for i in 1:length(its)]

## Convergence plot
gradnorms = [norm(its[i][4],h.meshes[end]) for i in 1:length(its)]
ϵs = [its[i][9] for i in 1:length(its)]
pp.newfig(1);
pp.semilogy(1:length(its), gradnorms)
pp.semilogy(1:length(its), ϵs)

## Reevaluation using new samples
s2 = SamplingData(1,n)
∇Js2 = [c(its[i][1],s2)[2] for i in 1:length(its)]
[norm(∇Js2[i],h.meshes[end]) for i in length(∇Js2)]
pp.semilogy(1:length(its),[norm(∇J,h.meshes[end]) for ∇J in ∇Js2])

## Evaluation of the cost function and its gradient on a line in the input space.

#J∇J - function yielding (J,∇J) pair
#u - input value
#d - direction in which to compute. If empty, chooses gradient in u
#s - step length
#n - number of evaluation points
function compute_line(J∇J::Function,u,d,s,n)
    "Computing line"
    steps = range(0, length = n, stop = 2s)
    J∇Js = []
    for i = 1:length(steps)
        push!(J∇Js, J∇J(u+d*steps[i]))
        println("step $i of $n")
    end
    Js = [t[1] for t in J∇Js]
    ∇Js = [t[2] for t in J∇Js]
    return steps, Js, ∇Js
end
compute_line(f::Function,u,s,n) = compute_line(f,u,-f(u)[2],s,n)

"Choose iteration number to plot the line for, and the number of evaluation points"
iteration_index = 11
n = 20

it = its[iteration_index]
s = it[10]
J∇J = u->c(u,s)
u = it[2]
d = it[7]
step = it[8]
steps, Js, ∇Js = compute_line(J∇J,u,d,step,n)

β = d/norm(d)
∇Jds = dot.(∇Js, [β])

pp.newfig(1)
pp.plot(steps,Js)
pp.newfig(2)
pp.plot(steps,∇Jds)

problematic_u = u+step*d
