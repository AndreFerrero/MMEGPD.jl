function _get_cdf_val(
    target_x,
    x_prev,
    theta,
    delta,
    norm_const,
    args...
)

    target_x <= 0 && return 0.0

    val, _ = quadgk(
        x -> megpd_joint(
            x,
            x_prev,
            theta,
            delta,
            args...
        ),
        0.0,
        target_x
    )

    return val / norm_const
end


# Standard EGPD
get_cdf_val(target_x, x_prev, theta, delta, norm_const) =
    _get_cdf_val(target_x, x_prev, theta, delta, norm_const)


# Bernstein EGPD
get_cdf_val(target_x, x_prev, theta, delta, norm_const, w) =
    _get_cdf_val(target_x, x_prev, theta, delta, norm_const, w)

function _sample_conditional(
    x_prev,
    theta,
    delta,
    args...
)

    # Normalising constant
    norm_const, _ = quadgk(
        x -> megpd_joint(
            x,
            x_prev,
            theta,
            delta,
            args...
        ),
        0.0,
        Inf
    )

    if !isfinite(norm_const) || norm_const <= 0.0
        throw(ArgumentError(
            "Invalid conditional normalising constant."
        ))
    end

    # Uniform draw
    p_target = rand()

    lower = 0.0
    upper = max(10 * x_prev, 1.0)

    f(x) = _get_cdf_val(
        x,
        x_prev,
        theta,
        delta,
        norm_const,
        args...
    ) - p_target

    # Expand interval until the root is bracketed
    while f(upper) < 0
        upper *= 2

        if !isfinite(upper)
            throw(ErrorException(
                "Could not bracket conditional quantile."
            ))
        end
    end

    return Roots.find_zero(
        f,
        (lower, upper),
        Roots.Bisection()
    )
end


# Standard EGPD
sample_conditional(x_prev, theta, delta) =
    _sample_conditional(x_prev, theta, delta)


# Bernstein EGPD
sample_conditional(x_prev, theta, delta, w) =
    _sample_conditional(x_prev, theta, delta, w)


# This is the main dispatcher function with the core logic
function _simulate_markov_megpd(
    n,
    theta,
    delta,
    args...;
    x0=1.0,
    burn_in_prop=0.0,
    show_progress::Bool=true
)

    n >= 1 || throw(ArgumentError("n must be positive."))

    0 <= burn_in_prop < 1 || throw(ArgumentError(
        "burn_in_prop must be in [0,1)."
    ))

    x = Vector{Float64}(undef, n)
    x[1] = x0

    p = show_progress ?
        Progress(n; desc="\n Simulating \n") : nothing

    for t in 2:n

        x[t] = sample_conditional(
            x[t-1],
            theta,
            delta,
            args...
        )

        show_progress && next!(p)

    end

    burn_in = floor(Int, n * burn_in_prop)

    return x[(burn_in+1):end]

end


# Standard EGPD
function simulate_markov_megpd(
    n,
    theta,
    delta;
    kwargs...
)

    return _simulate_markov_megpd(
        n, theta, delta;
        kwargs...
    )

end


# Bernstein EGPD
function simulate_markov_megpd(
    n,
    theta,
    delta,
    w;
    kwargs...
)

    return _simulate_markov_megpd(
        n, theta, delta, w;
        kwargs...
    )

end

function _multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    delta,
    args...;
    x0=nothing,
    burn_in_prop=0.0
)

    if isnothing(x0)
        x0 = fill(1.0, n_chains)
    end

    length(x0) == n_chains || throw(ArgumentError(
        "x0 must contain one initial value per chain."
    ))

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
            args...;
            x0=x0[i],
            burn_in_prop=burn_in_prop,
            show_progress=false
        )

        chains[:, i] = sim

    end

    return chains

end


# Standard EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    delta;
    kwargs...
)

    return _multiple_megpd_chains(
        n_chains, n, theta, delta;
        kwargs...
    )

end


# Bernstein EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    delta,
    w;
    kwargs...
)

    return _multiple_megpd_chains(
        n_chains, n, theta, delta, w;
        kwargs...
    )

end

# this is the dispatcher for the case where delta needs to be reconstructed from the r grid and the estimated delta
# Standard EGPD
function simulate_markov_megpd(
    n,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector;
    kwargs...
)

    delta_interp = build_delta_interp(
        r_grid,
        delta_hat
    )

    return simulate_markov_megpd(
        n,
        theta,
        delta_interp;
        kwargs...
    )

end


# Bernstein EGPD
function simulate_markov_megpd(
    n,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector,
    w;
    kwargs...
)

    delta_interp = build_delta_interp(
        r_grid,
        delta_hat
    )

    return simulate_markov_megpd(
        n,
        theta,
        delta_interp,
        w;
        kwargs...
    )

end

# Standard EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector;
    kwargs...
)

    delta_interp = build_delta_interp(
        r_grid,
        delta_hat
    )

    return multiple_megpd_chains(
        n_chains,
        n,
        theta,
        delta_interp;
        kwargs...
    )

end


# Bernstein EGPD
function multiple_megpd_chains(
    n_chains::Int,
    n::Int,
    theta,
    r_grid::AbstractVector,
    delta_hat::AbstractVector,
    w;
    kwargs...
)

    delta_interp = build_delta_interp(
        r_grid,
        delta_hat
    )

    return multiple_megpd_chains(
        n_chains,
        n,
        theta,
        delta_interp,
        w;
        kwargs...
    )

end

# this is the dispatcher for currently implemented delta functions organised by symbols
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

# Standard EGPD
function simulate_markov_megpd(
    n,
    theta,
    delta_name::Symbol;
    x0=1.0,
    burn_in_prop=0.0,
    show_progress::Bool=true,
    delta_kwargs...
)

    delta = _build_model_delta(
        theta,
        delta_name;
        delta_kwargs...
    )

    return simulate_markov_megpd(
        n,
        theta,
        delta;
        x0=x0,
        burn_in_prop=burn_in_prop,
        show_progress=show_progress
    )

end


# Bernstein EGPD
function simulate_markov_megpd(
    n,
    theta,
    delta_name::Symbol,
    w;
    x0=1.0,
    burn_in_prop=0.0,
    show_progress::Bool=true,
    delta_kwargs...
)

    delta = _build_model_delta(
        theta,
        delta_name;
        delta_kwargs...
    )

    return simulate_markov_megpd(
        n,
        theta,
        delta,
        w;
        x0=x0,
        burn_in_prop=burn_in_prop,
        show_progress=show_progress
    )

end

# Standard EGPD
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

    return multiple_megpd_chains(
        n_chains,
        n,
        theta,
        delta;
        x0=x0,
        burn_in_prop=burn_in_prop
    )

end


# Bernstein EGPD
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

    return multiple_megpd_chains(
        n_chains,
        n,
        theta,
        delta,
        w;
        x0=x0,
        burn_in_prop=burn_in_prop
    )

end