# Custom delta functions
function delta_exp(r) 0.2 + 0.6*exp(-r/5) end

function delta08(r) 0.8 end

# gd = Gamma(2, 10/3)
# function delta_gamma(r) 8 * pdf(gd, r) + 0.2 end

function build_delta_egpd_exp(;
    kappa,
    sigma,
    xi,
    c = 0.2,
    A = 0.6,
    p_ref = 0.95,
    a = 1.0
)
    0 < p_ref < 1 ||
        throw(ArgumentError("p_ref must be in (0, 1)"))
    a > 0 ||
        throw(ArgumentError("a must be positive"))
    c > 0 ||
        throw(ArgumentError("c must be positive"))
    A >= 0 ||
        throw(ArgumentError("A must be nonnegative"))

    # Characteristic radial scale
    q_ref = egpd_quantile(p_ref, kappa, sigma, xi)
    tau = q_ref / a

    return r -> c + A * exp(-r / tau)
end


# Dictionary entries are now factories.
# Each factory returns a function of r.
const DELTAS = Dict{Symbol, Function}(
    :delta08 => (() -> delta08),
    :delta_exp => (() -> delta_exp),
    :delta_egpd_exp => ((; kappa, sigma, xi, kwargs...) ->
        build_delta_egpd_exp(
            kappa = kappa,
            sigma = sigma,
            xi = xi;
            kwargs...
        ))
)

function get_delta(
    name::Symbol;
    kappa = nothing,
    sigma = nothing,
    xi = nothing,
    kwargs...
)
    factory = get(DELTAS, name) do
        throw(ArgumentError("Unknown delta function: $name"))
    end

    if name == :delta_egpd_exp
        isnothing(kappa) && throw(ArgumentError(
            "delta_egpd_exp requires kappa"
        ))
        isnothing(sigma) && throw(ArgumentError(
            "delta_egpd_exp requires sigma"
        ))
        isnothing(xi) && throw(ArgumentError(
            "delta_egpd_exp requires xi"
        ))

        return factory(;
            kappa = kappa,
            sigma = sigma,
            xi = xi,
            kwargs...
        )
    else
        return factory()
    end
end

# # Construct delta function from spline interpolation
# function build_delta_interp(r_grid, delta_hat)
#     itp = LinearInterpolation(r_grid, delta_hat, extrapolation_bc=Flat())
    
#     return r -> itp(r)
# end