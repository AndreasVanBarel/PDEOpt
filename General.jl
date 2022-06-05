"""
    General

Provides representation of grids ([`RegularGrid`](@ref)), grid points ([`Point`](@ref)) and grid hierarchies ([`Hierarchy`](@ref)). Allows mapping of data from one grid to another, implementing restrictions and prolongations, see [`lm`](@ref) and [`lm!`](@ref). Also provides utility such as calculating the volume spanned by the grid ([`nvolume`](@ref)), returning the grid boundary ([`boundary`](@ref)), calculating norms ([`norm`](@ref)) and inner products ([`inner`](@ref)) etc.

See also: [`getΔx`](@ref), [`getΔy`](@ref), [`inject`](@ref), [`extend`](@ref), [`extend!`](@ref), [`integral`](@ref)
"""
module General

export Point, Mesh, RegularGrid, RegularGrid1D, RegularGrid2D, Hierarchy
export getΔx, getΔy
export lm!, lm, inject # Level mapping
export extend, extend! # Hierarchy
export boundary
export nvolume, norm, inner, integral
export collect, apply
export DebugException, bottom

import Base: ==, +, -, *, /, \
import Base: show
import Base: iterate
import Base: getindex, size, length, ndims, eltype, collect, lastindex, axes
import Base: getproperty, propertynames
import LinearAlgebra: norm

using LinearAlgebra

##############
### POINTS ###
##############
"""
    Point

Representation for points of the domain.
"""
struct Point{N}
    coords::NTuple{N,Float64}
end
Point{N}(x...) where N = Point{N}(NTuple{N,Float64}(x))
Point(x...) = Point{length(x)}(x...)
Point(x) = Point{1}(tuple(x))
Point(x::Vector) = Point{length(x)}(tuple(x...))

function getproperty(p::Point{N}, s::Symbol) where N
    if s == :x && N>=1
        return p.coords[1]
    elseif s == :y && N>=2
        return p.coords[2]
    elseif s == :z && N>=3
        return p.coords[3]
    else
        return getfield(p,s)
    end
end
propertynames(p::Point{N}) where N = (:c,:x,:y,:z)[1:max(N+1,4)]

show(io::IO, p::Point) = print(io, "P$(p.coords)")
show(io::IO, p::Point{1}) = print(io, "P($(p.x))")
==(a::Point,b::Point) = a.coords == b.coords
+(a::Point, b::Point) = Point(a.coords .+ b.coords)
-(a::Point, b::Point) = Point(a.coords .- b.coords)
-(p::Point) = Point(.-p.coords)
*(p::Point, c::Number) = Point(p.coords .* c)
*(c::Number, p::Point) = *(p,c)
/(p::Point, c::Number) = Point(p.coords ./ c)
\(c::Number, p::Point) = /(p,c)
norm(a::Point, p::Real=2) = norm(collect(a.coords), p)

##############
### MESHES ###
##############
"""
    Mesh

Contains all information about the Mesh.
"""
abstract type Mesh end

"""
    RegularGrid

Rectilinear grid with equal spacing between the points.
"""
abstract type RegularGrid <: Mesh end

"""
    RegularGrid1D

1D Rectilinear grid with equal spacing between the points.
"""
struct RegularGrid1D <: RegularGrid
    nodes_x::Vector{Float64}
end
RegularGrid1D(mx::Int; start=0, fin=1) = RegularGrid1D(LinRange(start,fin,mx))

"""
    RegularGrid1D

2D Rectilinear grid with equal spacing between the points.
"""
struct RegularGrid2D <: RegularGrid
    nodes_x::Vector{Float64}
    nodes_y::Vector{Float64}
end
RegularGrid2D(mx::Int, my::Int; start::Vector=[0,0], fin::Vector=[1,1]) = RegularGrid2D(LinRange(start[1],fin[1],mx),LinRange(start[2],fin[2],my))
RegularGrid(mx::Int; start=0, fin=1) = RegularGrid1D(mx,start=start,fin=fin)
RegularGrid(mx::Int, my::Int; start::Vector=[0,0], fin::Vector=[1,1]) = RegularGrid2D(mx,my,start=start,fin=fin)

### Basic operations
@inline getΔx(m::RegularGrid) = (m.nodes_x[end]-m.nodes_x[1])/(length(m.nodes_x)-1)
@inline getΔy(m::RegularGrid2D) = (m.nodes_y[end]-m.nodes_y[1])/(length(m.nodes_y)-1)
ndims(m::RegularGrid1D) = 1
ndims(m::RegularGrid2D) = 2
eltype(::Type{RegularGrid1D}) = Point{1}
eltype(::Type{RegularGrid2D}) = Point{2}
size(m::RegularGrid1D) = (length(m.nodes_x),)
size(m::RegularGrid2D) = (length(m.nodes_x), length(m.nodes_y))
length(m::RegularGrid) = *(size(m)...)
lastindex(m::RegularGrid) = length(m)
lastindex(m::RegularGrid,d::Int) = size(m)[d]
axes(m::RegularGrid) = Base.OneTo.(size(m))

getindex(m::RegularGrid1D, i::Int) = Point(m.nodes_x[i])
getindex(m::RegularGrid2D, i::Int) = ((q,r) = divrem(i-1,length(m.nodes_x)); getindex(m,r+1,q+1))
getindex(m::RegularGrid2D, i::Int, j::Int) = Point(m.nodes_x[i], m.nodes_y[j])

==(m1::RegularGrid1D,m2::RegularGrid1D) = m1≡m2 || m1.nodes_x==m2.nodes_x
==(m1::RegularGrid2D,m2::RegularGrid2D) = m1≡m2 || m1.nodes_x==m2.nodes_x && m1.nodes_y==m2.nodes_y
equaldomain(m1::RegularGrid,m2::RegularGrid) = m1[1]==m2[1] && m1[end]==m2[end]
nvolume(m::RegularGrid1D) = abs(m.nodes_x[end]-m.nodes_x[1])
nvolume(m::RegularGrid2D) = abs((m.nodes_x[end]-m.nodes_x[1])*(m.nodes_y[end]-m.nodes_y[1]))

integral(a::Array{<:Number,N}, grid::RegularGrid) where N = sum(a)*nvolume(grid)/length(a)
integral(f::Function, grid::RegularGrid) = integral(f.(collect(grid)),grid)
inner(a::Vector{<:Number}, b::Vector{<:Number}, grid::RegularGrid1D) = (0.5*a[1]*b[1] + sum(view(a,2:length(a)-1).*view(b,2:length(b)-1)) + 0.5*a[end]*b[end])*nvolume(grid)/(length(a)-1)
function inner(a::Matrix{<:Number}, b::Matrix{<:Number}, grid::RegularGrid2D)
    corners = 0.25*(a[1,1]*b[1,1] + a[1,end]*b[1,end] + a[end,1]*b[end,1] + a[end,end]*b[end,end])
    sa1,sa2 = size(a)
    sb1,sb2 = size(b)
    edges = 0.5*(sum(view(a,2:sa1-1,1).*view(b,2:sb1-1,1)) +
                 sum(view(a,2:sa1-1,sa2).*view(b,2:sb1-1,sb2)) +
                 sum(view(a,1,2:sa2-1).*view(b,1,2:sb2-1)) +
                 sum(view(a,sa1,2:sa2-1).*view(b,sb1,2:sb2-1))
                )
    face = sum(view(a,2:sa1-1,2:sa2-1).*view(b,2:sb1-1,2:sb2-1))
    (corners + edges + face)*nvolume(grid)/((sa1-1)*(sa2-1))
end
norm(a::Vector{<:Number}, grid::RegularGrid1D) = sqrt(inner(a,a,grid))
norm(a::Matrix{<:Number}, grid::RegularGrid2D) = sqrt(inner(a,a,grid))
#Old and inaccurate
#norm(a::Array{<:Number,N}, grid::RegularGrid, p::Int=2) where N = isinf(p) ? max(abs.(a)) : norm(a,p)*(nvolume(grid)/length(a))^(1/p)
#inner(a::Array{<:Number,N}, b::Array{<:Number,N}, grid::RegularGrid) where N = sum(a.*b)*nvolume(grid)/length(a)

isinside(p::Point{1},m::RegularGrid1D) = m.nodes_x[1]≤p.x≤m.nodes_x[end]
isinside(p::Point{2},m::RegularGrid2D) = m.nodes_x[1]≤p.x≤m.nodes_x[end] && m.nodes_y[1]≤p.y≤m.nodes_y[end]
boundary(m::RegularGrid1D) = [m[1],m[end]]
function boundary(m::RegularGrid2D)
    vcat([m[i,1] for i in 1:length(m.nodes_x)],
    [m[end,i] for i in 2:length(m.nodes_y)],
    [m[i,end] for i in length(m.nodes_x)-1:-1:1],
    [m[1,i] for i in length(m.nodes_y)-1:-1:1])
end

### Iterator over the mesh
# state gives the indices of the last returned mesh point.
iterate(m::RegularGrid1D) = (m[1],1)
iterate(m::RegularGrid1D,state) = state>=length(m.nodes_x) ? nothing : (m[state+1], state+1)
iterate(m::RegularGrid2D) = (m[1,1],[1,1])
function iterate(m::RegularGrid2D,state)
    newstate = (state[1] == length(m.nodes_x) ? [1,state[2]+1] : state.+[1,0])
    newstate[2]>length(m.nodes_y) ? nothing : (m[newstate...], newstate)
end
collect(m::RegularGrid) = reshape([p for p in m],size(m))
apply(f::Function,m::Mesh) = reshape([f(p) for p in m],size(m))

### Printing
function show(io::IO, m::MIME"text/plain", mesh::RegularGrid1D)
    print(io, "$(length(mesh.nodes_x)) node $(typeof(mesh)) for [$(mesh.nodes_x[1]),$(mesh.nodes_x[end])]")
end
show(io::IO, m::Type{MIME"text/plain"}, mesh::RegularGrid1D) = print(io,"$(length(mesh.nodes_x)) nodes")
function show(io::IO, m::MIME"text/plain", mesh::RegularGrid2D)
    print(io, "$(length(mesh.nodes_x))×$(length(mesh.nodes_y)) $(typeof(mesh)) for [$(mesh.nodes_x[1]),$(mesh.nodes_x[end])]×[$(mesh.nodes_y[1]),$(mesh.nodes_y[end])]")
end
show(io::IO, m::Type{MIME"text/plain"}, mesh::Mesh) = print(io,"$(length(mesh.nodes_x))×$(length(mesh.nodes_y)) grid")
show(io::IO, mesh::RegularGrid) = show(io, MIME"text/plain", mesh)

### extending, i.e., padding
# Generates extended mesh containing all the previous points in addition to extra[d] more points in direction d
function extend(m::RegularGrid1D,extra::Tuple{Int})
    Δx = getΔx(m)
    nx = length(m.nodes_x) + extra[1]
    nodes_x = LinRange(m.nodes_x[1],m.nodes_x[end]+extra[1]*Δx,nx)
    RegularGrid1D(nodes_x)
end
function extend(m::RegularGrid2D,extra::Tuple{Int,Int})
    Δx = getΔx(m); Δy = getΔy(m)
    nx = length(m.nodes_x) + extra[1]
    ny = length(m.nodes_y) + extra[2]
    nodes_x = LinRange(m.nodes_x[1],m.nodes_x[end]+extra[1]*Δx,nx)
    nodes_y = LinRange(m.nodes_y[1],m.nodes_y[end]+extra[2]*Δy,ny)
    RegularGrid2D(nodes_x,nodes_y)
end
# Generates extended mesh containing all the previous points in addition to i more points in each direction
extend(m::RegularGrid,i::Int) = extend(m,ntuple(x->i,ndims(m)))

### refining
# Returns the same grid but with grid spacing halved.
function refine(m::RegularGrid1D,factor::Tuple{Int},offset::Tuple{Int})
    lx = length(m.nodes_x);
    nodes_x = LinRange(m.nodes_x[1], m.nodes_x[end], factor[1]*(lx-offset[1])+offset[1])
    RegularGrid1D(nodes_x)
end
function refine(m::RegularGrid2D,factor::Tuple{Int,Int},offset::Tuple{Int,Int})
    lx = length(m.nodes_x);
    nodes_x = LinRange(m.nodes_x[1], m.nodes_x[end], factor[1]*(lx-offset[1])+offset[1])
    ly = length(m.nodes_y);
    nodes_y = LinRange(m.nodes_y[1], m.nodes_y[end], factor[2]*(ly-offset[2])+offset[2])
    RegularGrid2D(nodes_x,nodes_y)
end

######################
### MESH HIERARCHY ###
######################
"""
    Hierarchy

Contains a number of RegularGrid objects.
"""
struct Hierarchy
    meshes::Vector{RegularGrid}
end
Hierarchy(mesh::Mesh, L::Int=0) = extend!(Hierarchy([mesh]), L)

# extends the maximum level to L such that L+1 levels in total are present.
function extend!(h::Hierarchy,L,factor::NTuple{N,Int},offset::NTuple{N,Int}=ntuple(x->1,N)) where  N
    for lvl in length(h.meshes)+1:L+1
        last = h.meshes[end]
        mesh = refine(last,factor,offset)
        push!(h.meshes,mesh)
    end
    return h
end
extend!(h::Hierarchy,L) = extend!(h,L,ntuple(x->2,ndims(h.meshes[end])),ntuple(x->1,ndims(h.meshes[end]))) #NOTE: Not type stable

#####################
### LEVEL MAPPING ###
#####################
### auxiliary functions
mask(m::Int) = [collect(1.0:2.0^m); collect(2.0^m-1.0:-1.0:1.0)]/4.0^m
mask(mx::Int, my::Int) = mask(mx)*mask(my)'
# interp: generates internal linear interpolation points, e.g., interp(1.0,3.0,3) = [1.5,2.0,2.5]
interp(start, stop, len::Int) = start.+(1:len)'.*(stop-start)./(len+1)

### level mapping functions
# overwrites vector v_new with vector v mapped to the dimensions of v_new.
function lm!(v::Vector{T}, v_new::Vector{T}) where T
    # trivial case
    if length(v) == length(v_new)
        v_new.=v
        return v_new
    end

    # checking sizes
    l = length(v)-1; l_new = length(v_new)-1;
    q = round(Int,log2(l_new/l)) # number of requested prolongations in x-direction
    if !(q < 0 ? l_new*2^-q == l : l*2^q == l_new)
        @error("Mapping from size $(size(v)) to $(size(v_new)) is not supported.")
    end

    if q < 0 # restriction
        # internal points
        m = mask(-q) # generation of mask
        for x in 2:l_new
            v_new[x] = sum(m.*v[2+(x-2)*2^(-q):x*2^(-q)]);
        end
        # end points
        v_new[1] = v[1]
        v_new[end] = v[end]
    else # interpolation
        for x in 1:l
            v_new[1+2^q*(x-1)] = v[x]; #copy existing points
            v_new[2+2^q*(x-1):2^q*x] =interp(v[x],v[x+1],2^q-1)
        end
        v_new[end] = v[end] #copy end point
    end
    return v_new
end
lm(v::Vector{T},n) where T = lm!(v,Vector{T}(undef,n))

# b is overwritten
function lm!(v::Matrix{T}, v_new::Matrix{T}) where T
    if size(v) == size(v_new); v_new.=v; return v_new; end

    ### Checking if meshes are compatible
    lx,ly = size(v).-1
    lx_new,ly_new = size(v_new).-1
    qx = round(Int,log2(lx_new/lx)) # number of requested prolongations in x-direction
    qy = round(Int,log2(ly_new/ly)) # number of requested prolongations in y-direction
    if !(qx < 0 ? lx_new*2^-qx == lx : lx*2^qx == lx_new) || !(qy < 0 ? ly_new*2^-qy == ly : ly*2^qy == ly_new)
        @error("Mapping from size $(size(v)) to $(size(v_new)) is not supported.")
    end

    function interp_x(qx::Int, qy::Int, v::Matrix{T})
        for col = 1:size(v,2)
            for i in 1:size(v,1)-1
                v_new[1+2^qx*(i-1),1+(col-1)*2^qy] = v[i,col]
                v_new[2+2^qx*(i-1):2^qx*i,1+(col-1)*2^qy] = interp(v[i,col],v[i+1,col],2^qx-1)'
            end
            v_new[end,1+(col-1)*2^qy] = v[end,col]
        end
    end
    function interp_y(qx::Int, qy::Int)
        for col = 1:2^qy:size(v_new,2)-1
            v_new[:,col+1:col+2^qy-1] = interp(view(v_new,:,col), view(v_new,:,col+2^qy), 2^qy-1)
        end
    end

    ### performing restrictions
    if qx < 0 || qy < 0
        mx = max(-qx,0); my = max(-qy,0)
        # internal points
        m = mask(mx,my) # generation of mask
        for x in 2:min(lx_new,lx), y in 2:min(ly_new,ly)
            v_new[1+(x-1)*2^max(0,qx),1+(y-1)*2^max(0,qy)] =
            sum(m.*v[2+(x-2)*2^mx:x*2^mx , 2+(y-2)*2^my:y*2^my]);
        end
        # boundary x (no corners)
        m = mask(mx)
        for x in 2:min(lx_new,lx)
            v_new[1+(x-1)*2^max(0,qx),1] = m ⋅ v[2+(x-2)*2^mx:x*2^mx,1];
            v_new[1+(x-1)*2^max(0,qx),end] = m ⋅ v[2+(x-2)*2^mx:x*2^mx,end];
        end
        # boundary y (no corners)
        m = mask(my)'
        for y in 2:min(ly_new,ly)
            v_new[1,1+(y-1)*2^max(0,qy)] = m ⋅ v[1,2+(y-2)*2^my:y*2^my];
            v_new[end,1+(y-1)*2^max(0,qy)] = m ⋅ v[end,2+(y-2)*2^my:y*2^my];
        end
        # corners
        v_new[1,1] = v[1,1]
        v_new[1,end] = v[1,end]
        v_new[end,1] = v[end,1]
        v_new[end,end] = v[end,end]
    end

    ### performing interpolations
    if qx > 0 interp_x(qx,max(qy,0), (qy<0 ? view(v_new,1:2^qx:lx_new,:) : v) ) end
    if qy > 0 interp_y(max(qx,0),qy) end

    return v_new
end
#lm(v::Matrix{T},Δℓ) where T = lm!(v,Matrix{T}(undef, (size(v).-1).*(2^Δℓ) ))
lm(v::Matrix{T},s1,s2) where T = lm!(v,Matrix{T}(undef,s1,s2))

# Simple restriction that just copies some values at regular intervals,
# and discards the values in between
function inject(v::Vector, ℓ::Int, ℓ_new::Int)
    ℓ_new <= ℓ || @error("Return level must be smaller than the original level.")
    q = 2^(ℓ-ℓ_new)
    v[1:q:end]
end
function inject(a::Matrix, ℓ::Int, ℓ_new::Int)
    ℓ_new <= ℓ || @error("Return level must be smaller than the original level.")
    q = 2^(ℓ-ℓ_new)
    a[1:q:end,1:q:end]
end

##################
### EXCEPTIONS ###
##################
struct DebugException <: Exception
    st::Vector{Base.StackTraces.StackFrame} #stacktrace
    text::String #information
    d #data
    e::Union{DebugException,Nothing} #a previous DebugException
end
DebugException() = DebugException(stacktrace(),"",nothing,nothing)
DebugException(text::String,d) = DebugException(stacktrace(),text,d,nothing)
DebugException(e::DebugException) = DebugException(stacktrace(),"",nothing,e)
DebugException(e::DebugException,text::String,d) = DebugException(stacktrace(),text,d,e)
bottom(e::DebugException) = isnothing(e.e) ? e : bottom(e.e)

end
