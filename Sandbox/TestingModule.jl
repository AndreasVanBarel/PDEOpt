module TestingModule

export tm
import Base: getproperty, propertynames, show
tm = TestingModule #short alias for this module

##############
function pad(A::AbstractArray{T,N}, n::Vararg{Int,N}) where {T,N}
    newsize = size(A).+n
    B = zeros(T,newsize...)
    B[axes(A)...] = A;
    return B
end
pad(A::AbstractArray{T,N}, n::Int) where {T,N} = pad(A, fill(n,N)...)
##############
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
propertynames(p::Point{N}) where N = [:coords,:x,:y,:z][1:min(N+1,4)]
show(io::IO, p::Point) = print(io, "P$(p.coords)")

end
