# run at start
# ensure that pwd() returns the path to the PDEOpt local repository
# To that effect, one may add a line such as
# cd("C:\\Users\\Andreas\\PDEOpt")
push!(LOAD_PATH, pwd())
push!(LOAD_PATH, pwd()*"\\Solvers", pwd()*"\\Sandbox")
Pkg.activate("./env")

using Statistics # stuff such as mean, var, cov, etc
using LinearAlgebra
using Printf
using JLD
using MAT
using Revise
import Plotter; pp = Plotter
using Utility
#using ProfileView

println("initalization complete.")
