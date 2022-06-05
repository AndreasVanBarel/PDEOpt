#############################
# Testing module MLMC       #
#############################
using MLMC
import Statistics: mean, var, cov
m = MLMC

#### SampleStats ####
stats = m.SampleStats(0.0,0)
for i in 1.0:3.0
    m.addsample!(stats,i)
end
# results should be 2, 0.66..., [0.66...]
mean(stats)
var(stats)
cov(stats)

stats = m.SampleStats(0.0,2)
for i in 1.0:10.0
    m.addsample!(stats,i)
end
# results should be 5.5, 8.25, [8.25; 3.75; 0.25]
mean(stats)
var(stats)
cov(stats)

#### mlmc ####
using Random
## simple scalar gensample function ##
gs(rℓ::Int, ℓ::Int, i::Int, n::Int) = (Random.seed!(abs(10i + ℓ)); rand() + 0.1*2.0^-rℓ) # deterministic gensample function
gs(rℓs::Vector{Int}, ℓ::Int, i::Int, n::Int) = [gs(rℓs[j],ℓ,i,n) for j in 1:length(rℓs)]
MLsamplestats = m.mlmc(gs, (x,ℓ)->x, 1, [10,1])[1]
mean(MLsamplestats)

## Matrix{Float64}-valued QoI ##
using General
using Stoch
using Volume_C_Solver
using Plotter; pp = Plotter

m0 = 5
nodes = LinRange(0,1,m0)
mesh = RegularGrid2D(nodes,nodes)
h = Hierarchy(mesh,2)
dist = LogNormal(x->0.0,exponentialcovariance(0.3,0.1,2))

seed = 1
samplers = [Stoch.gen_sampler(seed,dist,h.meshes[ℓ]) for ℓ=1:3]
function gs2(rℓ::Int,ℓ,i,n)
    s = samplers[ℓ+1](i)
    lm(s,(m0-1)*2^rℓ+1,(m0-1)*2^rℓ+1)
end
gs2(rℓs::Vector{Int},ℓ,i,n) = [gs2(rℓs[j],ℓ,i,n) for j in 1:length(rℓs)]
pp.surf(h.meshes[3], gs2(2,2,5,0),1)
pp.surf(h.meshes[2], gs2(1,2,5,0),2)
lm_s(s,ℓ) = General.lm(s,(m0-1)*2^ℓ+1,(m0-1)*2^ℓ+1)
## using mlmc
MLSS = m.mlmc(gs2, lm_s, 0, [10,5,1])[1]

pp.surf(h.meshes[1],mean(MLSS[1]),1)
pp.surf(h.meshes[2],mean(MLSS[2]),2)
pp.surf(h.meshes[3],mean(MLSS[3]),3)

## manually to check
samples1 = [gs2(0,0,i,0) for i in 1:10]
samples2 = [gs2(1,1,i,0) - lm_s(gs2(0,1,i,0),1) for i in 1:5]
samples3 = [gs2(2,2,i,0) - lm_s(gs2(1,2,i,0),2) for i in 1:1]
mean1 = sum(samples1)./10
mean2 = sum(samples2)./5
mean3 = sum(samples3)./1
pp.surf(h.meshes[1],mean1,11)
pp.surf(h.meshes[2],mean2,12)
pp.surf(h.meshes[3],mean3,13)

## checking RMSE
#get𝕍s(MLss,PointwiseRMSE())
getϵ(MLSS,PointwiseRMSE())
getϵ(MLSS,AggregateRMSE())

## adaptive mlmc
MLSS_ag = mlmc(gs2, lm_s, 0, 0.002, [1., 4., 16.], [16000, 4000, 1000]; rmsetype=AggregateRMSE())
MLSS_pw = mlmc(gs2, lm_s, 0, 0.002, [1., 4., 16.], [16000, 4000, 1000]; rmsetype=PointwiseRMSE())
