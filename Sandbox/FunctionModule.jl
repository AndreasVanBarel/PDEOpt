"""
    FunctionModule

This module describes so-called [`Fun`](@ref) objects, representing approximated functions. Internally they contain a mesh and an array of values. Methods are available to evaluate the function anywhere on its domain, e.g., by interpolation, and to perform other operations, such as integration.

The FunctionModule is not used in the remainder of PDEOpt, replaced by working with meshes and arrays of values separately.

"""
module FunctionModule

export Fun
export norm, dot, integral

import Base: ==, +, -, *, /, \, minimum, maximum, min, max
import Base: show
import Base: iterate
import Base.Broadcast: Broadcasted, broadcastable, BroadcastStyle, flatten
import LinearAlgebra: norm, dot
import General: integral

using General
#####################
### FUN{T} STRUCT ###
#####################
# This type represents an element in some function space.
# They can be evaluated in a Point. The result is of type T.
struct Fun{T}<:Function
    data::Array{T}
    mesh::Mesh
    function Fun{T}(data::Array{T}, mesh::Mesh) where T
        size(data) == size(mesh) || error("Fun cannot be constructed; data is incompatible with the mesh")
        new{T}(data,mesh)
    end
end

################################
### ADDITIONAL FUNCTIONALITY ###
################################

Fun{T}(data::AbstractArray, mesh::Mesh) where T = Fun{T}(Array{T}(data), mesh)
Fun(data::AbstractArray{T}, mesh::Mesh) where T = Fun{T}(data, mesh)
# constructor generating uninitialized Fun
Fun{T}(mesh::Mesh) where T = Fun{T}(Array{T,ndims(mesh)}(undef,length(mesh.nodes_x), length(mesh.nodes_y)), mesh)
Fun(mesh::Mesh) = Fun{Float64}(mesh)
# constructs a Fun{T} out of a function f:Point->T
Fun{T}(f::Function,mesh::Mesh) where T = Fun{T}(f.(collect(mesh)), mesh)
Fun(f::Function,mesh::Mesh) = Fun(f.(mesh), mesh) #f:T->Any
# constructs a Fun{T} out of another Fun{T} on a different mesh.
# Note that the values of Fun{T} are in this case simply either copied or interpolated linearly
Fun{T}(v::Fun,mesh::Mesh) where T = Fun{T}(v.(mesh),mesh) #NOTE: inefficient implementation
Fun(v::Fun{T},mesh::Mesh) where T = Fun{T}(v.(mesh),mesh) #NOTE: inefficient implementation
# constructs zero Fun{T} of the same type and size as the given Fun{T}
zero(v::Fun{T}) where T = Fun{T}(zeros(T, length(v.mesh.nodes_x), length(v.mesh.nodes_y)), v.mesh) #TODO: replace by zerovec type
#TODO: add zero(FT::Type{Fun{T}}) where T
function setboundary!(v::Fun{T}, f::Function) where T
    v.data[:,1].=f.(v.mesh[:,1])
    v.data[:,end].=f.(v.mesh[:,end])
    v.data[1,:].=f.(v.mesh[1,:])
    v.data[end,:].=f.(v.mesh[end,:])
end

show(io::IO, m::MIME"text/plain", v::Fun{T}) where T = show(io,v)
show(io::IO, m::MIME"text/html", v::Fun{T}) where T = show(io,v)
function show(io::IO, v::Fun{T}) where T
    print(io, "$(length(v.mesh.nodes_x))×$(length(v.mesh.nodes_y)) $(typeof(v))")
end

# domain and ranges
intype(v::Fun) = Point #note: domain of this type
outtype(v::Fun{T}) where T = T #note: range of this type

# Function to evaluate a Fun{T} in a point p
function (v::Fun{T})(p::Point) where T
    nx = v.mesh.nodes_x; ny = v.mesh.nodes_y;
    x = ceil(Int,(p.x-nx[1])/getΔx(v.mesh)) #first index in v.mesh (ceil because of 1-indexing)
    y = ceil(Int,(p.y-ny[1])/getΔy(v.mesh)) #second index in v.mesh (ceil because of 1-indexing)
    y1 = v.data[x,y] + (p.x - nx[x])/(nx[x+1] - nx[x]) * (v.data[x+1,y] - v.data[x,y]);
    y2 = v.data[x,y+1] + (p.x - nx[x])/(nx[x+1] - nx[x]) * (v.data[x+1,y+1] - v.data[x,y+1]);
    y1 + (p.y - ny[y])/(ny[y+1] - ny[y])*(y2-y1)
end
(v::Fun{T})(args...) where T = v(Point(args...))

# elementary functions (should be replaced by implementing broadcast methods, somewhat #TODO DEPRECATED)
+(a::Fun{T}, b::Fun{T}) where T = a.mesh == b.mesh ? Fun{T}(a.data + b.data, a.mesh) : error("Meshes not compatible")
-(a::Fun{T}, b::Fun{T}) where T = a.mesh == b.mesh ? Fun{T}(a.data - b.data, a.mesh) : error("Meshes not compatible")
-(a::Fun{T}) where T = Fun{T}(-a.data, a.mesh)
*(v::Fun{T}, c::Number) where T = Fun{T}(v.data*c, v.mesh)
*(c::Number, v::Fun{T}) where T = *(v,c)
*(a::Fun{T}, b::Fun{T}) where T = a.mesh == b.mesh ? Fun{T}(a.data .* b.data, a.mesh) : error("Meshes not compatible")
/(v::Fun{T}, c::Number) where T = Fun{T}(v.data/c, v.mesh)
/(c::Number, v::Fun{T}) where T = /(v,c)
/(a::Fun{T}, b::Fun{T}) where T = a.mesh == b.mesh ? Fun{T}(a.data ./ b.data, a.mesh) : error("Meshes not compatible")
#integral
integral(a::Any) = a
integral(a::Array{T}) where T = sum(a)
integral(v::Fun{T}) where T = sum(v.data)*area(v.mesh)/length(v.data)
#norm
norm(v::Fun{T}, p::Int=2) where T = scalednorm(v.data,p)
#inner product
dot(a::Fun{T}, b::Fun{T}) where T = a.mesh == b.mesh ? (a.data ⋅ b.data)*area(a.mesh)/length(a.data) : error("Meshes not compatible")
minimum(v::Fun{T}) where T = minimum(v.data) #calculates the minimum value in v
maximum(v::Fun{T}) where T = maximum(v.data) #calculates the maximum value in v
min(vs::Vararg{Fun{T}}) where T = all(v.mesh==vs[1].mesh for v in vs) ? Fun{T}(min.((v.data for v in vs)...),vs[1].mesh) : error("Meshes not compatible") #returns a Vec{T} that has at any point the minimum value of all input Vec{T}s at that point
max(vs::Vararg{Fun{T}}) where T = all(v.mesh==vs[1].mesh for v in vs) ? Fun{T}(max.((v.data for v in vs)...),vs[1].mesh) : error("Meshes not compatible") #returns a Vec{T} that has at any point the minimum value of all input Vec{T}s at that point

##############################
### BROADCASTING OF FUN{T} ###
##############################
# NOTE: new broadcasting code (v1.0.0)
# NOTE: currently in place broadcasting (using .=) is not supported!
# Essentially a Fun{T} object that allows indexing (used for easier broadcasting implementation)
struct FunWrapper{T}
    data::Array{T}
    mesh::Mesh
end
Base.size(fw::FunWrapper{T}) where T = size(fw.data) # if size is defined, then axes() has a fallback definition based on this
Base.getindex(fw::FunWrapper{T}, inds...) where T = fw.data[inds...]
Base.setindex!(fw::FunWrapper{T}, val, inds...) where T = fw.data[inds...] = val
broadcastable(v::Fun{T}) where T = FunWrapper{T}(v.data,v.mesh)
struct FunStyle <: BroadcastStyle end
BroadcastStyle(::Type{<:FunWrapper}) = FunStyle()
BroadcastStyle(::FunStyle, ::BroadcastStyle) = FunStyle()
BroadcastStyle(::BroadcastStyle, ::FunStyle) = FunStyle()
function copy(bc::Broadcasted{FunStyle})
    f, args, mesh = resolve(bc)
    Fun(broadcast(f, args...), mesh)
end
function resolve(bc::Broadcasted{FunStyle})
    DEBUG = false;
    flat = flatten(bc)
    args = flat.args
    args_new = []
    DEBUG && println("args:\n $args")
    first = true;
    local mesh
    for i in eachindex(args)
        arg = args[i]
        if typeof(arg)<:FunWrapper
            first ? (mesh=arg.mesh; first=false) : mesh==arg.mesh || error("Meshes not compatible")
            push!(args_new, arg.data)
        else
            push!(args_new, arg)
        end
    end
    DEBUG && println("new args:\n $args_new")
    return flat.f, args_new, mesh
end

##############################
### EXTENDED FUNCTIONALITY ###
##############################

# lm! - levelmap: Maps Fun v to level of Fun v_new
# assumes compatibility of meshes.
# overwrites v_new (second argument)
function lm!(v::Fun{T}, v_new::Fun{T}) where T
    ### Trivial case
    if v.mesh == v_new.mesh; v_new.data[:] = v.data[:]; return v_new; end

    ### Checking if meshes are compatible
    lx = length(v.mesh.nodes_x)-1; lx_new = length(v_new.mesh.nodes_x)-1;
    qx = round(Int,log2(lx_new/lx)) # number of requested prolongations in x-direction
    ly = length(v.mesh.nodes_y)-1; ly_new = length(v_new.mesh.nodes_y)-1;
    qy = round(Int,log2(ly_new/ly)) # number of requested prolongations in y-direction
    #println("qx=$qx, qy=$qy")
    if !(qx < 0 ? lx_new*2^-qx == lx : lx*2^qx == lx_new) || !(qy < 0 ? ly_new*2^-qy == ly : ly*2^qy == ly_new)
        @error("Mapping between these representations not supported")
    end

    function interp_x(qx::Int, qy::Int, v::AbstractArray{Float64,2})
        for col = 1:size(v,2)
            for i in 1:size(v,1)-1
                v_new.data[1+2^qx*(i-1),1+(col-1)*2^qy] = v[i,col]
                v_new.data[2+2^qx*(i-1):2^qx*i,1+(col-1)*2^qy] = interp(v[i,col],v[i+1,col],2^qx-1)'
            end
            v_new.data[end,1+(col-1)*2^qy] = v[end,col]
        end
    end
    function interp_y(qx::Int, qy::Int)
        for col = 1:2^qy:size(v_new.data,2)-1
            v_new.data[:,col+1:col+2^qy-1] = interp(view(v_new.data,:,col), view(v_new.data,:,col+2^qy), 2^qy-1)
        end
    end

    ### performing restrictions
    if qx < 0 || qy < 0
        mx = max(-qx,0); my = max(-qy,0)
        # internal points
        m = mask(mx,my) # generation of mask
        for x in 2:min(lx_new,lx), y in 2:min(ly_new,ly)
            v_new.data[1+(x-1)*2^max(0,qx),1+(y-1)*2^max(0,qy)] =
            sum(m.*v.data[2+(x-2)*2^mx:x*2^mx , 2+(y-2)*2^my:y*2^my]);
        end
        # boundary x (no corners)
        m = mask(mx)
        for x in 2:min(lx_new,lx)
            v_new.data[1+(x-1)*2^max(0,qx),1] = m ⋅ v.data[2+(x-2)*2^mx:x*2^mx,1];
            v_new.data[1+(x-1)*2^max(0,qx),end] = m ⋅ v.data[2+(x-2)*2^mx:x*2^mx,end];
        end
        # boundary y (no corners)
        m = mask(my)'
        for y in 2:min(ly_new,ly)
            v_new.data[1,1+(y-1)*2^max(0,qy)] = m ⋅ v.data[1,2+(y-2)*2^my:y*2^my];
            v_new.data[end,1+(y-1)*2^max(0,qy)] = m ⋅ v.data[end,2+(y-2)*2^my:y*2^my];
        end
        # corners
        v_new.data[1,1] = v.data[1,1]
        v_new.data[1,end] = v.data[1,end]
        v_new.data[end,1] = v.data[end,1]
        v_new.data[end,end] = v.data[end,end]
    end

    ### performing interpolations
    if qx > 0 interp_x(qx,max(qy,0), (qy<0 ? view(v_new.data,1:2^qx:lx_new,:) : v.data) ) end
    if qy > 0 interp_y(max(qx,0),qy) end

    return v_new
end
# lm - levelmap: Maps function v to given mesh
lm(v::Fun{T}, mesh::Mesh) where T = lm!(v, Fun(mesh))
lm(v::Fun{T}, v_new::Fun{T}) where T = lm(v, v_new.mesh)

function inject(v::Fun{T}, mesh::Mesh) where T
    v.mesh==mesh && return v #trivial case
    equaldomain(v.mesh, mesh) || error("Domain of v and given mesh do not correspond")
    lx = length(v.mesh.nodes_x) -1; lx_new = length(mesh.nodes_x) -1;
    ly = length(v.mesh.nodes_y) -1; ly_new = length(mesh.nodes_y) -1;
    lx%lx_new==0 && ly%ly_new==0 || error("given mesh is not a submesh")
    qx=lx÷lx_new; qy=ly÷ly_new
    Fun{T}(v.data[1:qx:end,1:qy:end],mesh)
end

end
