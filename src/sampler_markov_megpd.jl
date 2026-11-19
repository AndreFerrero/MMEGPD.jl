using QuadGK
using Roots
using ProgressMeter
using Random

# Assumes the following are defined elsewhere (include this file after them):
#   megpd_joint(x, x_prev, theta, delta)        -- standard EGPD
#   megpd_joint(x, x_prev, theta, delta, w)     -- Bernstein EGPD
#   build_delta_interp(r_grid, delta_hat)
#   get_delta(name::Symbol; kappa, sigma, xi, kwargs...)


# =============================================================================
# 1. Model dispatch: standard vs Bernstein (w === nothing means standard)
# =============================================================================

_joint(x, y, theta, delta, ::Nothing) = megpd_joint(x, y, theta, delta)
_joint(x, y, theta, delta, w)         = megpd_joint(x, y, theta, delta, w)


# =============================================================================
# 2. Unnormalised densities (returned as closures x -> density)
#    - conditional: density of X_t given X_{t-1} = x_prev
#    - marginal   : density of X_t, joint integrated over the conditioning value
# =============================================================================

_density(x_prev::Real, theta, delta, w) =
    x -> _joint(x, x_prev, theta, delta, w)

_density(theta, delta, w) =
    x -> first(quadgk(y -> _joint(x, y, theta, delta, w), 0.0, Inf))


# =============================================================================
# 3. Generic machinery shared by conditional and marginal:
#    normalising constant, CDF, inversion sampling
# =============================================================================

function _normalising_constant(dens)

    norm_const, _ = quadgk(dens, 0.0, Inf)

    if !isfinite(norm_const) || norm_const <= 0.0
        throw(ArgumentError("Invalid normalising constant."))
    end

    return norm_const
end


function _cdf_value(dens, norm_const, target_x)

    target_x <= 0 && return 0.0

    val, _ = quadgk(dens, 0.0, target_x)

    return val / norm_const
end


function _sample_by_inversion(dens, norm_const, upper0)

    p_target = rand()

    lower = 0.0
    upper = upper0

    f(x) = _cdf_value(dens, norm_const, x) - p_target

    # Expand interval until the root is bracketed
    while f(upper) < 0
        upper *= 2

        if !isfinite(upper)
            throw(ErrorException("Could not bracket quantile."))
        end
    end

    return Roots.find_zero(
        f,
        (lower, upper),
        Roots.Bisection()
    )
end


# =============================================================================
# 4. Conditional and marginal sampling / CDF
#    (standard EGPD: no w;  Bernstein EGPD: with w)
# =============================================================================

# --- conditional CDF (kept for backward compatibility) ---
get_cdf_val(target_x, x_prev, theta, delta, norm_const) =
    _cdf_value(_density(x_prev, theta, delta, nothing), norm_const, target_x)

get_cdf_val(target_x, x_prev, theta, delta, norm_const, w) =
    _cdf_value(_density(x_prev, theta, delta, w), norm_const, target_x)


# --- conditional sampling ---
function _sample_conditional(x_prev, theta, delta, w)

    dens = _density(x_prev, theta, delta, w)
    norm_const = _normalising_constant(dens)

    return _sample_by_inversion(dens, norm_const, max(10 * x_prev, 1.0))
end

sample_conditional(x_prev, theta, delta) =
    _sample_conditional(x_prev, theta, delta, nothing)

sample_conditional(x_prev, theta, delta, w) =
    _sample_conditional(x_prev, theta, delta, w)


# --- marginal sampling: n_draws independent draws, normalising constant
#     computed once (it is a nested integral, so this matters) ---
function _sample_marginal(n_draws::Int, theta, delta, w)

    n_draws >= 1 || throw(ArgumentError("n_draws must be positive."))

    dens = _density(theta, delta, w)
    
    # The marginal is a proper density, so it integrates to 1 by construction:
    # no need to compute the (nested) normalising constant numerically.
    norm_const = 1.0

    draws = Vector{Float64}(undef, n_draws)

    Threads.@threads for i in 1:n_draws
        draws[i] = _sample_by_inversion(dens, norm_const, 1.0)
    end

    return draws
end

sample_marginal(n_draws::Int, theta, delta) =
    _sample_marginal(n_draws, theta, delta, nothing)

sample_marginal(n_draws::Int, theta, delta, w) =
    _sample_marginal(n_draws, theta, delta, w)

# single draw
sample_marginal(theta, delta) =
    _sample_marginal(1, theta, delta, nothing)[1]

sample_marginal(theta, delta, w) =
    _sample_marginal(1, theta, delta, w)[1]


# =============================================================================
# 5. Starting values: nothing -> draw from the marginal, Real -> use as is
# =============================================================================

function _check_x0(x0::Real)
    x0 > 0 || throw(ArgumentError("x0 must be strictly positive."))
    return Float64(x0)
end

# single chain
_resolve_x0(::Nothing, theta, delta, w) = _sample_marginal(1, theta, delta, w)[1]
_resolve_x0(x0::Real,  theta, delta, w) = _check_x0(x0)

# several chains
_resolve_x0(::Nothing, n_chains::Int, theta, delta, w) =
    _sample_marginal(n_chains, theta, delta, w)

_resolve_x0(x0::Real, n_chains::Int, theta, delta, w) =
    fill(_check_x0(x0), n_chains)

function _resolve_x0(x0::AbstractVector, n_chains::Int, theta, delta, w)

    length(x0) == n_chains || throw(ArgumentError(
        "x0 must contain one initial value per chain."
    ))

    return Float64[_check_x0(v) for v in x0]
end


# =============================================================================
# 6. Core simulators (w === nothing -> standard EGPD, otherwise Bernstein)
# =============================================================================

function _simulate_markov_megpd(
    n,
    theta,
    delta,
    w;
    x0=nothing,
    burn_in_prop=0.0,
    show_progress::Bool=true
)

    n >= 1 || throw(ArgumentError("n must be positive."))

    0 <= burn_in_prop < 1 || throw(ArgumentError(
        "burn_in_prop must be in [0,1)."
    ))

    x = Vector{Float64}(undef, n)
    x[1] = _resolve_x0(x0, theta, delta, w)

    p = show_progress ?
        Progress(n; desc="\n Simulating \n") : nothing

    for t in 2:n

        x[t] = _sample_conditional(x[t-1], theta, delta, w)

        show_progress && next!(p)

    end

    burn_in = floor(Int, n * burn_in_prop)

    return x[(burn_in+1):end]

end


function _multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    delta,
    w;
    x0=nothing,
    burn_in_prop=0.0
)

    # Resolved up-front so random starts share one marginal normalising constant
    x0s = _resolve_x0(x0, n_chains, theta, delta, w)

    burn_in = floor(Int, n * burn_in_prop)
    chain_length = n - burn_in

    chains = Matrix{Float64}(
        undef,
        chain_length,
        n_chains
    )

    Threads.@threads for i in 1:n_chains

        sim = _simulate_markov_megpd(
            n,
            theta,
            delta,
            w;
            x0=x0s[i],
            burn_in_prop=burn_in_prop,
            show_progress=false
        )

        chains[:, i] = sim

    end

    return chains

end


# =============================================================================
# 7. delta construction from a symbol
# =============================================================================

function _build_model_delta(
    theta,
    delta_name::Symbol;
    kwargs...
)

    if length(theta) == 3

        kappa = theta[1]
        sigma = theta[2]
        xi = theta[3]

    elseif length(theta) == 2

        kappa = theta[1]
        sigma = 1.0
        xi = theta[2]

    else

        throw(ArgumentError(
            "theta must have length 2 (Bernstein) " *
            "or 3 (standard EGPD)."
        ))

    end

    return get_delta(
        delta_name;
        kappa=kappa,
        sigma=sigma,
        xi=xi,
        kwargs...
    )

end


# =============================================================================
# 8. Public API: single chain
#    x0 defaults to nothing -> random start drawn from the marginal
# =============================================================================

# Ready-made delta: standard EGPD
simulate_markov_megpd(n, theta, delta; kwargs...) =
    _simulate_markov_megpd(n, theta, delta, nothing; kwargs...)

# Ready-made delta: Bernstein EGPD
simulate_markov_megpd(n, theta, delta, w; kwargs...) =
    _simulate_markov_megpd(n, theta, delta, w; kwargs...)


# delta reconstructed from r grid and estimated delta: standard EGPD
function simulate_markov_megpd(
    n,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector;
    kwargs...
)

    delta_interp = build_delta_interp(r_grid, delta_hat)

    return _simulate_markov_megpd(n, theta, delta_interp, nothing; kwargs...)

end

# delta reconstructed from r grid and estimated delta: Bernstein EGPD
function simulate_markov_megpd(
    n,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector,
    w;
    kwargs...
)

    delta_interp = build_delta_interp(r_grid, delta_hat)

    return _simulate_markov_megpd(n, theta, delta_interp, w; kwargs...)

end


# delta given by name: standard EGPD
function simulate_markov_megpd(
    n,
    theta,
    delta_name::Symbol;
    x0=nothing,
    burn_in_prop=0.0,
    show_progress::Bool=true,
    delta_kwargs...
)

    delta = _build_model_delta(
        theta,
        delta_name;
        delta_kwargs...
    )

    return _simulate_markov_megpd(
        n,
        theta,
        delta,
        nothing;
        x0=x0,
        burn_in_prop=burn_in_prop,
        show_progress=show_progress
    )

end

# delta given by name: Bernstein EGPD
function simulate_markov_megpd(
    n,
    theta,
    delta_name::Symbol,
    w;
    x0=nothing,
    burn_in_prop=0.0,
    show_progress::Bool=true,
    delta_kwargs...
)

    delta = _build_model_delta(
        theta,
        delta_name;
        delta_kwargs...
    )

    return _simulate_markov_megpd(
        n,
        theta,
        delta,
        w;
        x0=x0,
        burn_in_prop=burn_in_prop,
        show_progress=show_progress
    )

end


# =============================================================================
# 9. Public API: multiple chains
#    x0: nothing (independent random starts, default), a Real (same start for
#    all chains) or a vector (one start per chain)
# =============================================================================

# Ready-made delta: standard EGPD
multiple_megpd_chains(n_chains::Int, n::Int, theta, delta; kwargs...) =
    _multiple_megpd_chains(n_chains, n, theta, delta, nothing; kwargs...)

# Ready-made delta: Bernstein EGPD
multiple_megpd_chains(n_chains::Int, n::Int, theta, delta, w; kwargs...) =
    _multiple_megpd_chains(n_chains, n, theta, delta, w; kwargs...)


# delta reconstructed from r grid and estimated delta: standard EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector;
    kwargs...
)

    delta_interp = build_delta_interp(r_grid, delta_hat)

    return _multiple_megpd_chains(
        n_chains, n, theta, delta_interp, nothing; kwargs...
    )

end

# delta reconstructed from r grid and estimated delta: Bernstein EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector,
    w;
    kwargs...
)

    delta_interp = build_delta_interp(r_grid, delta_hat)

    return _multiple_megpd_chains(
        n_chains, n, theta, delta_interp, w; kwargs...
    )

end


# delta given by name: standard EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    delta_name::Symbol;
    x0=nothing,
    burn_in_prop=0.0,
    delta_kwargs...
)

    delta = _build_model_delta(
        theta,
        delta_name;
        delta_kwargs...
    )

    return _multiple_megpd_chains(
        n_chains,
        n,
        theta,
        delta,
        nothing;
        x0=x0,
        burn_in_prop=burn_in_prop
    )

end

# delta given by name: Bernstein EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    delta_name::Symbol,
    w;
    x0=nothing,
    burn_in_prop=0.0,
    delta_kwargs...
)

    delta = _build_model_delta(
        theta,
        delta_name;
        delta_kwargs...
    )

    return _multiple_megpd_chains(
        n_chains,
        n,
        theta,
        delta,
        w;
        x0=x0,
        burn_in_prop=burn_in_prop
    )

end