using Optimization

####### NORMAL SINGLE LEVEL NCG #######
L = 0
problem = problems[5]
h = Hierarchy(problem)
extend!(h,L) # sets maximum level at L
compute = ComputeStruct(problem,h)
u = Fun(x->0.0, h.meshes[end]) # Starting value for u given on the finest level

its = []; #to save the intermediate results in
@time u = ncg(u,compute,5e-6,20; save=its)

# fixed set of samples
n=[8,7,6,5]
#n=[7211,2877,378]
n=[11978, 1429, 332, 76]
n = [1000]
samplingdata = SamplingData(0,n)
@time u = ncg(u,compute,0.0,10,samplingdata; save=its)

samplingdata = SamplingData(0,[25])
@time u = ncg(u,compute,0.0,30,samplingdata)

# using Profile
# Profile.clear()
# s = SamplingData(0,n)
# @profile compute(u,s)
# @profile ncg(u,compute,0.0,1,n,true)
# @profile u = ncg(u,0.01,10,false)
