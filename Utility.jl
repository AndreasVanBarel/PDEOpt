module Utility

export resize_string, inspect, pause, toMatlab, tikzprint

using General
using MAT

## Quick inspection of objects
# truncate or append the string to a certain number of characters.
function resize_string(s::String, n, separator::String="..")
    if length(s) <= n
        s=s*repeat(" ", n-length(s))
    else
        start_fraction = 0.65
        start_n = floor(Int,(n-2)*start_fraction)
        end_n = n-length(separator)-start_n
        s = s[1:start_n]*separator*s[end-end_n+1:end]
    end
    return s
end
function inspect(t::Union{Tuple,Vector})
    type_length = 20
    data_length = 58
    println("i   Type"*repeat(" ",type_length-2)*"Data")
    for i in 1:length(t)
        el = t[i]
        index_string = resize_string("$i",4)
        type_string = resize_string("$(typeof(el))",type_length," .. ")
        data_string = resize_string("$el", data_length)
        println(index_string*type_string*"  "*data_string)
    end
end

## Pause until keypress
function pause(message::String="Paused. Press enter to continue (or space+enter+enter if at >julia prompt), or type \"abort\" to attempt to abort by throwing an error.")
    println(message)
    r = readline(stdin)
    if r=="abort"
        @error("This error attempts to abort the program. Generated by the user.")
    end
    return nothing
end

## Exporting to Matlab
function toMatlab(data::Dict{String,<:Any}, name::String="data")
	path = "C:\\Users\\Andreas\\Documents\\MATLAB\\imported\\"
	matwrite(path*name*".mat", data)
end
function toMatlab(grid::Mesh, values, name::String="surf")
	toMatlab(Dict(
	"values" => values,
	"nodes1" => grid.nodes_x,
	"nodes2" => grid.nodes_y
	), name=name)
end

## Exporting to Tikz
function tikzprint(u::Vector, option = "Int", start = 1)
    if option == "0to1"
		tikzprint(collect(LinRange(0,1,length(u))),u)
    else option == "Int"
		tikzprint(start:start+length(u)-1,u)
    end
end
function tikzprint(nodes::Vector, values::Vector)
	for i = 1:length(nodes)
		println(nodes[i], "  ", values[i], " \\\\")
	end
end

end
