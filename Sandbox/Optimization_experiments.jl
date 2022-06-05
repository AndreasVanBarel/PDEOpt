## EXPERIMENT 3 ##
# Checks how many samples must be retained on the coarser MG/OPT levels

K = 3 # maximum level
problem = problems[1]
h = Hierarchy(problem, K)
LM = gen_LM(h) #level mapping function

n = [1000,200,40,8]
qstart=1.0/256; qend= 1; qlen = 9
qs = exp10.(range(log10(qstart); stop=log10(qend), length=qlen))

# Executing MG/OPT optimization for different values of q
results = [];
for i in 1:qlen
    q = qs[i]
    println("Retaining $(100q)% of samples on each coarser level")
    ss = [SamplingData(0, ceil.(Int, n[1:ℓ+1]*q^(K-ℓ))) for ℓ in 0:K]
    f = gen_f(ss, problem, h)

    # executing MG/OPT optimization
    u = Fun(x->0.0, h.meshes[end]) # Starting value for u given on the finest level
    us = [u]; gnorms = [norm(f(u,K)[2])]; ηs = [NaN]
    for i=1:25
        println("V-cycle $i")
        u,gnorm,η = mgoptV(u,zero(u),f,LM,K,K; print=false)
        push!(us,u); push!(gnorms,gnorm); push!(ηs,η)
    end
    push!(results,(q,us,gnorms,ηs))
end

##
res = results[:]

for i in 1:length(results)
    r = results[i]
    res[i] = (r[1], r[3], r[4])
end

pp.newfig(1)
for i in 1:length(res)
    r = res[i]; q = r[1]; gnorms = r[2]
    pp.semilogy(1:length(gnorms),gnorms)
end


## EXPERIMENT 3 (OLD)##
# Checks how many samples must be retained on the coarser MG/OPT levels
L = 2 # maximum level
problem = problems[4]
h = Hierarchy(problem, L)
LM = gen_LM(h)
nL = [40000, 8000, 1300]

qstart=1e-2; qend= 1e-1; qlen = 6
qs = exp10.(range(log10(qstart); stop=log10(qend), length=qlen))

# costfunction evaluation at finest MG/OPT level (=MLMC with full samples)
function fL(u::Fun)
    c = ComputeStruct(problem, h)
    c(u,SamplingData(0,nL))
end

# part 1: Executing finest level optimization using smoother
println("Finest level optimization\n Iteration: ")
u = Fun(x->0.0, h.meshes[end]) # Starting value for u given on the finest level
its_fine = [(u,fL(u)...)]
for i=1:10
    global u
    print("=")
    u, ∇J₀ = smooth(u,u->fL(u)[1:2],2) # performs a smoothing step at the finest level
    push!(its_fine,(u,fL(u)...))
end

# part 2: Executing MG/OPT optimization for different values of q
results = [];
for i in 1:qlen
    q = qs[i]
    println("Retaining $(100q)% of samples on each coarser level")
    ss = [SamplingData(0, ceil.(Int, nL[1:ℓ+1]*q^(L-ℓ))) for ℓ in 0:L]
    f = gen_f(ss, problem, h)

    # executing MG/OPT optimization
    println("MG/OPT optimization")
    u = Fun(x->0.0, h.meshes[end]) # Starting value for u given on the finest level
    its_V = [(u,fL(u)...)]
    for i=1:5
        u = mgoptV(u,zero(u),f,LM,L,L)[1] # performs a V-cycle
        push!(its_V,(u,fL(u)...))
    end
    push!(results,(q,its_V))
end

# Plotting
# part 1
pp.newfig(11)
norms_fine = [norm(its_fine[i][2]) for i in 1:length(its_fine)]
pp.semilogy(norms_fine)

# part 2
pp.newfig(12)
for r in 1:length(results)
    q = results[r][1]
    its_V = results[r][2]
    norms_V = [norm(its_V[i][3]) for i in 1:length(its_V)]
    pp.semilogy(norms_V);
end
pp.legend([[string(round(q,digits=2)) for q in qs]...], loc = "lower left")
pp.figure(12)

# Saving
using JLD
save("opt_exp1.jld", "its_fine", getdata(its_fine), "results", getdata(results))


#its_V = results[1][2]
#ϵV = [its_V[i][5] for i in 1:length(its_V)]
#pp.semilogy(ϵV, ls="dashed", color="black")
#pp.legend(["Fine", [string(round(ps[i],2)) for i in 1:10]..., "ϵ"], loc = "lower left")

####################################
## PAPER 1 Optimization algorithm ##
####################################
#Robust optimization on the finest level only

"ASSUME:"
# compute - A ComputeStruct (see Gradient.~)
# u0      - A compatible starting value for the iteration.

using Gradient
using Optimization

its = []
q = 0.5
τ = 1e-4
it = 10
u = ncg(u0, c, τ, it, q; print=2, plots=false, save=its, norm=scalednorm)

# Convergence plot
gradnorms = [scalednorm(its[i][3]) for i in 1:length(its)]
ϵs = [its[i][6] for i in 1:length(its)]
pp.newfig(1);
pp.semilogy(1:length(its), gradnorms)
pp.semilogy(1:length(its), ϵs./q)

# Solution plot
J,∇J,𝔼y,𝕍y,s = compute(u,τ*q)
pp.newfig(2); pp.surf(u)
pp.newfig(3); pp.surf(∇J)

####################################
## PAPER 2 Optimization algorithm ##
####################################
using General
using Optimization

"ASSUME:"
# cs      - A ComputeStruct (see Gradient.~)
# u0      - A compatible starting value for the iteration.
# lm_mgopt-

K = prob.L
rmsetype = MLMC.AggregateRMSE()

function rob_opt(u,cs::Vector{<:Function},K,rmsetype,τ::Real,r::Real,ϵ::Real,imax::Int)
    its = Array{Any,2}(undef,0,7)
    println("Starting Robust Optimization.")
    q = 1.0/16
    for i = 1:imax
        # TESTING ϵ = r*τ
        # constructing the function f (one additional gradient evaluation; #TODO: make more efficient)
        println("Determining number of samples for accuracy ϵ = $ϵ")
        _,∇J,_,_,s = cs[end](u,ϵ) #J,∇J,𝔼y,𝕍y,s
        ss = [SamplingData(s.seed, ceil.(Int, s.n[1:ℓ+1]*q^(K-ℓ))) for ℓ in 0:K]
        f = gen_f(cs,ss)
        println("n = $(s.n); ||∇J|| = $(norm(∇J))")

        # running MG/OPT V-cycle
        println("Running MG/OPT V-cycle")
        t = @elapsed u,gnorm,η = mgoptV(u,zero(u),f,lm_mgopt,K,K; print=true)
        η = min(η,0.5)
        println("gnorm = $gnorm, η=$η")
        its = [its; [i, ϵ, s.n, gnorm, η, t, NaN]']

        # testing convergence
        if gnorm ≤ τ
            gnorm_new = norm(compute(u,r*τ)[2])
            its[end,end] = gnorm_new
            @printf("Testing ||∇J|| for convergence. Fixed samples: %8.6e; New samples: %8.6e\n", gnorm, gnorm_new)
            gnorm_new ≤ τ && return u, its
        end

        # new RMSE
        ϵ = max(r*τ,r*η*gnorm)
    end
    return u, its
end

@time u, its = rob_opt(u0,cs,K,rmsetype,2e-4, 0.25, 1e-1, 10)

# Generating data to save
