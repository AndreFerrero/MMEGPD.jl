# Custom delta functions
function delta_strong_upper(r) 0.2 + 0.6*exp(-r/5) end

function delta08(r) 0.8 end

# gd = Gamma(2, 10/3)
# function delta_gamma(r) 8 * pdf(gd, r) + 0.2 end

const DELTAS = Dict(
    :delta08 => delta08,
    :delta_strong_upper => delta_strong_upper,
)

function get_delta(name::Symbol)
    get(DELTAS, name) do
        throw(ArgumentError("Unknown delta function: $name"))
    end
end

# # Construct delta function from spline interpolation
# function build_delta_interp(r_grid, delta_hat)
#     itp = LinearInterpolation(r_grid, delta_hat, extrapolation_bc=Flat())
    
#     return r -> itp(r)
# end
