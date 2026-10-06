module MMEGPD

# ─────────────────────────────────────────────────────────────
# Standard library — no installation required
# ─────────────────────────────────────────────────────────────
using Random
using LinearAlgebra
using Statistics
using Base.Threads


# ─────────────────────────────────────────────────────────────
# Packages — need to be installed
# ─────────────────────────────────────────────────────────────
using Distributions
using Interpolations
using BSplines
using QuadGK
using Roots
using ProgressMeter
using SpecialFunctions
using Optim
using Optimisers
using Plots
using HCubature

using ADTypes: AutoForwardDiff
import ForwardDiff

# CORE FUNCTIONS
include("delta.jl")
include("egpd.jl")
include("megpd_joint.jl")
include("megpd_cond.jl")
include("megpd_marginal.jl")
include("as_pairs.jl")

include("sampler_markov_megpd.jl")

# SPLINES RELATED FUNCTIONS
# BASIS OBJECTS
include("BSpline_basis.jl")
include("CRSpline_basis.jl")
# SPLINE CONSTRUCTOR
include("delta_spline.jl")
# P-Splines penalties
include("penalties.jl")
# Plotting from fit object
include("plot_spline.jl")

# MODEL RELATED FUNCTIONS (PMLE, Turing)
include("markov_megpd_loglik.jl")
include("fit_markov_megpd_splines.jl")

# LAMBDA CHOICE FUNCTIONS
# Laplace approximation
include("laplace.jl")


end