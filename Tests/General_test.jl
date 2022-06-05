#############################
# Testing module General    #
#############################

using General
#using Plotter

##############
### POINTS ###
##############
p1 = Point(1,2,3)
p1.y
p1+p1

##############
### MESHES ###
##############
nodes_x = LinRange(0.5,1,3)
nodes_y = LinRange(-1,2,5)
grid1d = RegularGrid1D(nodes_x)
grid2d = RegularGrid2D(nodes_x,nodes_y)
[x for x in grid1d]
[x for x in grid2d]
collect(grid1d)
collect(grid2d)
nvolume(grid1d)
nvolume(grid2d)
integral(x->1,grid1d)
integral(x->1,grid2d)
integral(p->p.x^2,grid1d)
integral(p->p.x^2,grid2d)
boundary(grid1d)
boundary(grid2d)
extend(grid1d,(1,))
extend(grid2d,(1,2))

#################
### HIERARCHY ###
#################
h1 = Hierarchy(grid1d,3)
h2 = Hierarchy(grid2d,3)

#################
### LEVEL MAP ###
#################
vs5 = [Matrix{Float64}(I,5,5)[:,i] for i in 1:5]
vs9 = [Matrix{Float64}(I,9,9)[:,i] for i in 1:9]
m95 = hcat(lm.(vs5,9)...)
m59 = hcat(lm.(vs9,5)...)

r5 = rand(5)
r9 = rand(9)
inner(lm(r5,9),r9,RegularGrid1D(9))
inner(r5,lm(r9,5),RegularGrid1D(5))

# coarsening
zfun = (x...)->Float64(all(0.25.<=x.<=0.75)) # target function
m = 17
grid = RegularGrid1D(m)
z = zfun.(grid.nodes_x)
lm(z,9)
lm(z,5)
lm(lm(z,9),5)
