## EXPERIMENT 2 ##
# NCG method as described in the first paper #

using Gradient
using Optimization

"ASSUME:
 c       - A ComputeStruct (see Gradient.~)
 u0      - A compatible starting value for the iteration."

## Experiment parameters
maxit = 200 #12
r = 0.5 #ratio of RMSE w.r.t. gradient norm
 # Experiment
its2 = []
@time u = ncg(u0, c, 1e-4, maxit, r; norm=norm_∇J, print=2, save=its2, ls_options=ls_options, Π=prob.Π)
s = SamplingData(12345,its2[end][9].n)

# its2[1] contains the state before any NCG step has been performed.
# its2[i+1] contains the information regarding NCG step i and the state after it.
# length(its2) equals the number of NCG steps + 1.
# Structure of its2 is as follows
#1   u      typeof(u)
#2   J      Float64
#3   ∇J     typeof(u)
#4   𝔼y     typeof(𝔼y)
#5   𝕍y     typeof(𝔼y)
#6   d      typeof(u)       # Search direction, equal to nothing for its[1]
#7   step   Float64         # Step size, equal to nothing for its[1]
#8   ϵ      Float64
#9   s      SamplingData
#10	 t 		Float64			# Equal to 0.0 for its[1]

## Convergence plot
gradnorms = [norm_∇J(its2[i][3]) for i in 1:length(its2)]
ϵs = [its2[i][8] for i in 1:length(its2)]
pp.newfig(1);
pp.semilogy(1:length(its2), gradnorms)
pp.semilogy(1:length(its2), ϵs)

## Full reevaluation using new samples
s2 = SamplingData(1,its2[end][9].n)
∇Js2 = [c(its2[i][1],s2)[2] for i in 1:length(its2)]
pp.semilogy(1:length(its2),[norm_∇J(∇J) for ∇J in ∇Js2])

## Quick reevaluation using new samples
interval = 10 # reevaluates only once every 10 iterations
s2 = SamplingData(1000,its2[end][9].n)
@time ∇Js2 = [c(its2[i][1],s2)[2] for i in 1:interval:length(its2)]
pp.semilogy(1:interval:length(its2),[norm_∇J(∇J) for ∇J in ∇Js2])

## Saving its2
save("exp2.jld", "its", its2)

## Loading its2
path = "Data\\exp_DN\\p4\\"
its2 = load(path*"exp2.jld")["its"]
u = its2[end][1]
s = SamplingData(12345,its2[end][9].n)

## Constructing table from its2
function construct_table(print=true)
	text = String[]
	for i = 1:length(its2)
		# Check whether new samples were taken in this iteration
		i_s = "$(i) & "
		ϵ = its2[i][8]
		ϵ_s = @sprintf "\\num{%4.2e} & " ϵ
		n = its2[i][9].n
		n_s = *(string.(n).*" & "...)
		J = its2[i][2]
		J_s = @sprintf "\\num{%4.2e} & " J
		grad = norm_∇J(its2[i][3])
		grad_s = @sprintf "\\num{%4.2e} & " grad
		solves = MLMC.nb_solves(n,costs)*2
		solves_s = @sprintf("%2.0f & ", solves)
		time_s = @sprintf("%2.0f ", its2[i][10])
		end_s = "\\\\"
		line = i_s*ϵ_s*n_s*J_s*grad_s*solves_s*time_s*end_s
		push!(text,line)
		print && println(line)
	end
	return text
end
construct_table()

## Getting the equivalent fine level solves
