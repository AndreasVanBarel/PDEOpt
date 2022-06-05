module Plotter
# This module extends PyPlot functoins and defines some new ones.
# Note that the extended methods are not explicitly exported. Rather, it is necessary to do
# using PyPlot or import PyPlot. Alternatively, one can access the functions using their
# fully qualified names, e.g., Plotter.surf.

using General
using FunctionModule
using PyPlot

export usetex
export newfig, set_cmap, get_cmap
export ticksize, labelsize, allsize, remove_background
export rotate_preset
# extending methods requires either fully qualified name in function header, or an import (e.g.) PyPlot: surf.

# NOTE: pygui() shows the Python backend used for plotting

# global colormap
# setcmap(x::String) = global colormap = ColorMap(x)
# setcmap(x::ColorMap) = global colormap = x
# getcmap() = colormap
# setcmap("viridis")

#function __init__()
#    rc("font",family="serif",serif="Computer Modern")
#    rc("text",usetex=true)
#end

usetex(b::Bool) = rc("text",usetex=b)

newfig(fig) = (f = figure(fig); clf(); return f)

### SURF ###
# making surf work with Vectors as the first two arguments
function PyPlot.surf(x::Vector, y::Vector, zz::Matrix, v...)
    xx=x.*ones(1,length(y))
    yy=ones(length(x)).*y'
    fig = surf(xx,yy,zz,v...)
    ax = gca()
    ax.set_proj_type("ortho")
    background(false)
    return fig
end
# surf on a given mesh
function PyPlot.surf(grid::RegularGrid2D, data::Matrix)
    #rc("font",family="serif",serif="Computer Modern")
    mx,my = size(data)
    fig = surf(reshape(grid.nodes_x,(mx,1)), reshape(grid.nodes_y,(1,my)), data, cmap=get_cmap(), antialiased=true, alpha=1.0, rstride=1, cstride=1, edgecolor="black", linewidth=0.1)
    xlabel(raw"$x_1$"); ylabel(raw"$x_2$")
    ax = gca()
    ax.set_proj_type("ortho")
    background(false)
    ax.set_xlim([grid[1].x,grid[end].x])
    ax.set_ylim([grid[1].y,grid[end].y])
    [t.set_va("center") for t in ax.get_xticklabels()]
    [t.set_ha("right") for t in ax.get_xticklabels()]
    [t.set_va("center") for t in ax.get_yticklabels()]
    [t.set_ha("left") for t in ax.get_yticklabels()]
    [t.set_va("center") for t in ax.get_zticklabels()]
    [t.set_ha("left") for t in ax.get_zticklabels()]
    #ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    #ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    #ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    #ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    #ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    #ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4
    tight_layout()
    return fig
end
PyPlot.surf(mesh::Mesh, data::Array, fig) = (newfig(fig); surf(mesh,data))
# surf a Fun object
PyPlot.surf(v::Fun) = surf(v.mesh, v.data)
PyPlot.surf(v::Fun,fig) = (newfig(fig); surf(v))

### PLOT ###

# plot on a given mesh
function PyPlot.plot(grid::RegularGrid2D, data::Matrix)
    #rc("font",family="serif",serif="Computer Modern")
    nx = grid.nodes_x;
    ny = grid.nodes_y;
    ext = [nx[1],nx[end],ny[1],ny[end]]
    #asp = (ny[end]-ny[1])/(nx[end]-nx[1])
    fig = imshow(rotl90(data), extent=ext) #aspect=asp)
    xlabel(raw"$x_1$"); ylabel(raw"$x_2$")
    return fig
end
function PyPlot.plot(grid::RegularGrid1D, data::Vector)
    #rc("font",family="serif",serif="Computer Modern")
    PyPlot.plot(grid.nodes_x,data)
    ax = gca()
    ax.set_xlim([grid[1].x,grid[end].x])
end
PyPlot.plot(mesh::Mesh, data::Array, fig) = (newfig(fig); plot(mesh,data))

# plot a Fun object
PyPlot.plot(v::Fun) = surf(v.mesh, v.data)
PyPlot.plot(v::Fun,fig) = (newfig(fig); surf(v))

function ticksize(s)
    axes = PyPlot.gcf().axes
    for ax in axes
        ax.tick_params(axis = "both", which = "major", labelsize = s)
    end
end
function labelsize(s)
    axes = PyPlot.gcf().axes
    for ax in axes
        ax.xaxis.label.set_fontsize(s)
        ax.yaxis.label.set_fontsize(s)
        try ax.zaxis.label.set_fontsize(s) catch KeyError end
    end
end
allsize(s) = (ticksize(s); labelsize(s))
scientific_notation_x(d::Int=2) = gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.$(d)e"))
scientific_notation_y(d::Int=2) = gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.$(d)e"))
scientific_notation_z(d::Int=2) = gca().zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.$(d)e"))
scientific_colorbar(d::Int=2) = colorbar(format="%.$(d)e")

function background(b::Bool=true)
    ax = PyPlot.gca()
    ax.xaxis.pane.fill = b
    ax.yaxis.pane.fill = b
    ax.zaxis.pane.fill = b
end

# Changes the rotation of a 3D plot to a different preset
function rotate_preset()
    ax = PyPlot.gca()
    ax.view_init(30,60)
    [t.set_va("center") for t in ax.get_xticklabels()]
    [t.set_ha("left") for t in ax.get_xticklabels()]
    [t.set_va("center") for t in ax.get_yticklabels()]
    [t.set_ha("right") for t in ax.get_yticklabels()]
    [t.set_va("center") for t in ax.get_zticklabels()]
    [t.set_ha("right") for t in ax.get_zticklabels()]
end

function force_update(plt=PyPlot.gcf())
    plt.set_figheight(plt.get_figheight())
end

# Useful commands:
# tight_layout()
# cb.ax.tick_params(labelsize=17) # for cb a colorbar object

end
