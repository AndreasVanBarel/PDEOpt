## EXPERIMENT 4_W ##
# Paper 2 MG/OPT experiments #
# Using W-cycles instead of V-cycles #
using Gradient
using Optimization

"ASSUME:
 c        - A ComputeStruct (see Gradient.~)
 u0       - A compatible starting value for the iteration, given at the finest level
 prob     - A Problem<x> object
 lm_mgopt - A level mapping function for MG/OPT
 norm_∇J  - A function calculating the norm of the gradient giving comparable results at all levels. "

K = prob.L # depth of the MG/OPT iteration

# q - amount of samples to retain on a courser level
# r - maximum value of RMSE(∇J)/norm(∇J)
function robust_mgopt(u,c,τ,K,ϵ,imax;r=1.0,q=1/16,save=false,smoother,inner_∇J,norm_∇J,print=2)
	μ_pre = [fill(2,prob.L)...,0];
	μ_post = [fill(2,prob.L)...,4];

	η = NaN
    for i = 1:imax+1
        print>1 && i<=imax && println("Robust MG/OPT: i = $i, ϵ = $ϵ, η = $η.")

        # Generate new sample set and therefore new f
        print>1 && i<=imax && @printf("Generating new sample set... ")
        J,∇J,𝔼y,𝕍y,s = c(u,ϵ)
        print>1 && i<=imax && println("n = $(s.n).")
        ss = [SamplingData(s.seed+ℓ-K, ceil.(Int, s.n[1:ℓ+1]*q^(K-ℓ))) for ℓ in 0:K]
        f = gen_f(cs, ss)

		# saving iteration data
		save!=false && push!(save,[u,J,∇J,𝔼y,𝕍y,ϵ,s,nothing,nothing,0.0])

        # stopping condition
        print>1 && @printf("Current gradient: ‖∇J‖ = %8.6e. \n", norm_∇J(∇J))
        if norm_∇J(∇J) <= τ
            return u
        end
        i>imax && break

        # V-cycle
		result = @timed(mgoptW(u,zero(u),f,lm_mgopt,K,μ_pre,μ_post; smooth=smoother, print=print, inner=inner_∇J, ls_options=ls_options_mgopt))
        u,j,g,j_start,g_start = result[1] #note j_start = J and g_start = ∇J
		time = result[2]
		gnorm = norm_∇J(g); gnorm_start = norm_∇J(g_start)
		η = min(0.5,gnorm/gnorm_start)

		# updating iteration data
		if save!=false
			save[i][8] = j
			save[i][9] = g
			save[i][10] = time
		end

        # new value for ϵ
        ϵ = max(r*τ,r*η*gnorm)
    end
    print>1 && println("No convergence.")
    return u
end

its4 = []
@time u = robust_mgopt(u0,c,5e-5,K,1e-1,10; r=0.5,q=1/16, save=its4,smoother=smoother,norm_∇J=norm_∇J,inner_∇J=inner_∇J)
s = SamplingData(12345,its4[end][7].n)

# its4[i] contains information regarding the state before and after Vcycle i.
# its4[end] contains the state after the last Vcycle, evaluated using new samples.
# length(its4) equals the number of Vcycles + 1.
# Structure of its4 is as follows
#1   u      typeof(u)
#2   J      Float64
#3   ∇J     typeof(u)
#4   𝔼y     typeof(𝔼y)
#5   𝕍y     typeof(𝔼y)
#6   ϵ      Float64
#7   s      SamplingData
#8   j 		Float64
#9   g      typeof(u)		# Equal to nothing for its[end]
#10	 t 		Float64			# Equal to 0.0 for its[end]

## Convergence plot
gnorms = [norm_∇J(its4[i][3]) for i in 1:length(its4)]
pp.newfig(7); pp.semilogy(1:length(its4), gnorms)

## Saving its4
save("exp4.jld", "its", its4)

## Loading its4
path = "Data\\exp_B\\p16\\"
its4 = load(path*"exp4.jld")["its"]
u = its4[end][1]
s = SamplingData(12345,its4[end][7].n)

## Getting equivalent fine level solves
grad_per_smooth = 2
grad_per_update = 1
function nb_solves(n,costs,μ_pre,μ_post,grad_per_smooth::Int,grad_per_update::Int,q=1/16)
	n_k = [ceil.(Int, n[1:k+1]*q^(K-k)) for k in 0:K]
	costs_k = [costs[1:k+1] for k in 0:K]
	Cg_k = [MLMC.nb_solves(n_k[k+1],costs_k[k+1])*costs[k+1]/costs[K+1] for k in 0:K] # number of solves for a gradient call at MG/OPT lvl k
	Cs_k = grad_per_smooth.*Cg_k
	solves_smooth_k = (μ_pre+μ_post).*Cs_k # Smoothing costs
	solves_update_k = grad_per_update.*Cg_k # Level update costs
	solves_V = sum(solves_smooth_k.+solves_update_k)
	return solves_V, solves_smooth_k, solves_update_k
end
t1,t2,t3 = nb_solves(its4[end][7].n,costs,μ_pre,μ_post,grad_per_smooth,grad_per_update)

## Constructing table from its4 (run above first)
function construct_table(print=true)
	text = String[]
	for i = 1:length(its4)
		i_s = "$(i) & "
		ϵ = its4[i][6]
		ϵ_s = @sprintf "\\num{%4.2e} & " ϵ
		n = its4[i][7].n
		n_s = *(string.(n).*" & "...)
		J_start = its4[i][2]
		J_start_s = @sprintf "\\num{%4.2e} & " J_start
		grad_start = norm_∇J(its4[i][3])
		grad_start_s = @sprintf "\\num{%4.2e} & " grad_start
		J_end = its4[i][8]
		J_end_s = i!=length(its4) ? @sprintf("\\num{%4.2e} & ",J_end) : " & "
		grad_end_s = i!=length(its4) ? @sprintf("\\num{%4.2e} & ", norm_∇J(its4[i][9])) : " & "
		solves = nb_solves(n,costs,μ_pre,μ_post,grad_per_smooth,grad_per_update)[1]
		solves_s = @sprintf("%2.0f & ", solves)
		time_s = @sprintf("%2.0f", its4[i][10])
		end_s = "\\\\"
		line = i_s*ϵ_s*n_s*J_start_s*J_end_s*grad_start_s*grad_end_s*solves_s*time_s*end_s
		push!(text,line)
		print && println(line)
	end
end
construct_table()
