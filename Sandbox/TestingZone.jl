s = SamplingData(0,[64,16,4,1])
@enter c(u,s)

##
T = Float64
nx = size(k,1)
y = Array{T}(undef,nx,nt)
y_prime = Array{T}(undef,nx,nt)
solve_state!(y,u,k,s,Δx,Δt;y_prime=y_prime,WARN_STAB=false) # sets y and yprime
J = 0.5/(nx-1)*(sum((y[:,end]-z).^2) + α*sum(u.^2))
p = solve_adjoint(y,y_prime,u,z,k,s,Δx,2/5*Δt) # gets p
∇J = -p[:,1] .+ α.*u
##

# Generation of Quasi Monte Carlo points using a simple Lattice Rule.
s = 3 #number of dimensions
N = 17
z = [1,3,5]
P = mod.(z.*(0:N-1)',N)/N
pp.clf()
pp.scatter(P[1,:],P[2,:],zs=P[3,:])
pp.plot3D(P[1,:],P[2,:],P[3,:], ".")

k=2
function fv(u)
    J₀,∇J₀ = f(u,k)
    Jv = J₀ - vc⋅u/length(u)
    ∇Jv = ∇J₀ - vc
    return Jv,∇Jv
end

function tikzprint(u::Vector, option = "Int", start = 0)
    if option == "0to1"
        for i = 1:length(u)
            println((i-1)/(length(u)-1),"  ", u[i], " \\\\")
        end
    else option == "Int"
        for i = 1:length(u)
            println(i+start-1, "  ", u[i], " \\\\")
        end
    end
end

# truncate or append the string to a certain number of characters.
function resize_string(s::String, n)
    if length(s) <= n
        s=s*repeat(" ", n-length(s))
    else
        start_fraction = 0.65
        start_n = floor(Int,(n-2)*start_fraction)
        end_n = n-2-start_n
        s = s[1:start_n]*".."*s[end-end_n+1:end]
    end
    return s
end
function inspect(t::Tuple)
    type_length = 20
    data_length = 58
    println("i   Type"*repeat(" ",type_length-2)*"Data")
    for i in 1:length(t)
        el = t[i]
        index_string = resize_string("$i",4)
        type_string = resize_string("$(typeof(el))",type_length)
        data_string = resize_string("$el", data_length)
        println(index_string*type_string*"  "*data_string)
    end
end

#= NOTES
Needed figures:
1) DN convergence of NCG on different discretization grids (single figure). In fact, this figure is also nice for V-control. (2 figures)
2) Plot showing the degeneration if not enough samples are retained on the coarser level.
3)

=#

## Currying and function piping
import Base: |, getindex, getproperty

# piping
|(x,f::Function) = f(x)

# currying
struct Curry end
~ = Curry() #WARNING: Occludes bitwise negation!
getindex(f::Function) = f
getindex(f::Function,x) = f(x)
getindex(f::Function,::Curry) = t->f(t)
getindex(f::Function,x,args...) = ((s...)->f(x,s...))[args...]
getindex(f::Function,::Curry,args...) = (λ(t,s...) = f[t,args...](s...); λ(t) = f[t,args...]; λ)
(::Curry)(args...) = (λ(f,s...)=f[args...](s...); λ(f)=f[args...]; λ)
getproperty(::Curry,s::Symbol) = x->getproperty(x,s)
getindex(::Curry,args...) = (λ(x,s...)=getindex[x,args...](s...);λ(x)=getindex[x,args...];λ)

f(a,b,c,d) = 1000a + 100b + 10c + d
t = f[1,~,3,~]
t(5,6)

t = ~(9)
t(sqrt)
t(-)
t = ~(~)
t(sqrt,4)

t = (~).foo
struct Foo; foo::Int; end; foo = Foo(2); t(foo)

vec = [1,4,8,16,32]
t = (~)[5]
t(vec)
t = (~)[~]
t(vec,4)

3 | plus[~,5] | f[1,2,~,4] | log10[~]

t = sin[cos[~]]


## Generation of some figures for a presentation or paper.
# Generation of surface plot figure
fontsize = 14;
pp.newfig(11); pp.clf()
pp.surf(mesh,data);
pp.allsize(fontsize);
pp.scientific_notation_z(0)

# Generation of flat figure with colorbar
fontsize = 16;
pp.newfig(10);
pp.plot(mesh,data);
pp.allsize(fontsize);
cb = pp.scientific_colorbar(0)
cb.ax.tick_params(labelsize=fontsize)
pp.tight_layout()

## Animation of a given y using Plots
using Plots

plt = plot(
    xlim = (0, 1),
    ylim = (-0.8, 1.1),
    title = "Burgers' Equation",
    marker = 0.1,
)

grid = mesh
nx = grid.nodes_x
plot!(plt,nx,y[:,1])
savefig(plt,"test.png")

z = prob.zfun.(nx)

anim = Animation()
println("Generating animation...")
for t=1:40:size(y,2)
    time = @sprintf("%4.2f",t/size(y,2))
    plt = plot(nx,[y[:,t],z],
    xlim = (0, 1),
    ylim = (-0.8, 1.1),
    xtickfont = font(14),
    ytickfont = font(14),
    xlabel = "x",
    ylabel = "y",
    guidefont = font(14),
    title = "t=$time", lw=3, leg=false)
    frame(anim)
end
gif(anim)
