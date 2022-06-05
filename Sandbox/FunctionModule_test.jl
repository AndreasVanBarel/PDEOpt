using FunctionModule
using General

n_c = 5
n_f = 9
mesh_c = RegularGrid2D(n_c,n_c)
mesh_f = RegularGrid2D(n_f,n_f)
mesh_fc = RegularGrid2D(n_f,n_c)
mesh_cf = RegularGrid2D(n_c,n_f)

f = p->sin(4p.x-6p.y^2)-3p.x
v_c = Fun{Float64}(f, mesh_c)
v_f = Fun{Float64}(f, mesh_f)
v_fc = Fun{Float64}(f, mesh_fc)
v_cf = Fun{Float64}(f, mesh_cf)

# testing Fun evaluation in a DomainPoint
nodes_x = LinRange(0,2,10)
nodes_y = LinRange(0,1,30)
mesh = RegularGrid2D(nodes_x,nodes_y)
data = [x^2+2y^2 for x in nodes_x, y in nodes_y]
fun = Fun(data,mesh)
p = Point(0.5,0.2)
fun(p) #0.340...

# testing broadcasting
# ToDo: doesn't work anymore since broadcasting architecture got changed in some Julia update. Not critical, ignore the error.
nodes = 1:3
mesh = RegularGrid2D(nodes,nodes)
v1 = Fun([10i+j for i in 1:3, j in 1:3], mesh)
v2 = Fun([i*j for i in 1:3, j in 1:3], mesh)
# experiment using v1 and v2, e.g.
norm( (v1.*v2).data - v1.data.*v2.data)
