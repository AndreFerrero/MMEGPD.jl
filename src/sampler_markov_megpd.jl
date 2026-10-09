function get_cdf_val(
    target_x,
    x_prev,
    kappa,
    sigma,
    xi,
    delta,
    norm_const,
)

    if target_x <= 0
        return 0.0
    end

    val, _ = quadgk(
        x -> megpd_joint(
            x,
            x_prev,
            kappa,
            sigma,
            xi,
            delta
        ),
        0.0,
        target_x,
    )

    return val / norm_const
end

function sample_conditional(
    x_prev,
    kappa,
    sigma,
    xi,
    delta
)

    # Normalizing constant
    norm_const, _ = quadgk(
        x -> megpd_joint(
            x,
            x_prev,
            kappa,
            sigma,
            xi,
            delta
        ),
        0.0,
        Inf,
    )

    # Uniform draw
    p_target = rand()

    lower = 1e-10
    upper = max(10*x_prev, 1.0)

    f(x) = get_cdf_val(
        x,
        x_prev,
        kappa,
        sigma,
        xi,
        delta,
        norm_const,
    ) - p_target

    # Expand search interval until it brackets the root
    while f(upper) < 0
        upper *= 2
    end

    return Roots.find_zero(f, (lower, upper), Roots.Bisection())
end


# This is the main dispatcher function with the core logic
function simulate_markov_megpd(
    n_steps,
    kappa,
    sigma,
    xi,
    delta;
    x0=1.0,
    burn_in_prop=0.0,
    show_progress::Bool=true
)

    0 <= burn_in_prop < 1 || throw(ArgumentError(
        "burn_in_prop must be in [0,1)."
    ))

    x = Vector{Float64}(undef, n_steps)
    x[1] = x0

    p = show_progress ? Progress(n_steps; desc="\n Simulating \n") : nothing

    for t in 2:n_steps
        x[t] = sample_conditional(
            x[t-1],
            kappa,
            sigma,
            xi,
            delta
        )

        show_progress && next!(p)
    end
    
    burn_in = floor(Int, n_steps * burn_in_prop)
    final_chain = x[(burn_in+1):end]

    return final_chain
end

function multiple_megpd_chains(
    n_chains::Int,
    n_steps::Int,
    kappa,
    sigma,
    xi,
    delta;
    x0=1.0,
    burn_in_prop=0.0
)

    burn_in = floor(Int, n_steps * burn_in_prop)
    chain_length = n_steps - burn_in

    chains = Matrix{Float64}(undef, chain_length, n_chains)

    Threads.@threads for i in 1:n_chains

        println("Simulation:", i)
        sim = simulate_markov_megpd(
            n_steps,
            kappa,
            sigma,
            xi,
            delta;
            x0=x0,
            burn_in_prop=burn_in_prop,
            show_progress=false
        )

        chains[:, i] = sim
    end

    return chains
end

# this is the dispatcher for the case where delta needs to be reconstructed from the r grid and the estimated delta
function simulate_markov_megpd(
    n_steps,
    kappa,
    sigma,
    xi,
    r_grid::AbstractVector,
    delta_hat::AbstractVector;
    x0=1.0,
    burn_in_prop=0.0,
    show_progress::Bool=true
)

    delta_interp = build_delta_interp(r_grid, delta_hat)

    return simulate_markov_megpd(
        n_steps,
        kappa,
        sigma,
        xi,
        delta_interp;
        x0=x0,
        burn_in_prop=burn_in_prop,
        show_progress=show_progress
    )
end

function multiple_megpd_chains(
    n_chains::Int,
    n_steps::Int,
    kappa,
    sigma,
    xi,
    r_grid::AbstractVector,
    delta_hat::AbstractVector;
    x0=1.0,
    burn_in_prop=0.0
)
    burn_in = floor(Int, n_steps * burn_in_prop)
    chain_length = n_steps - burn_in

    chains = Matrix{Float64}(undef, chain_length, n_chains)

    Threads.@threads for i in 1:n_chains
        println("Simulation:", i)
        sim = simulate_markov_megpd(
            n_steps,
            kappa,
            sigma,
            xi,
            r_grid,
            delta_hat;
            x0=x0,
            burn_in_prop=burn_in_prop,
            show_progress=false
        )

        chains[:, i] = sim
    end

    return chains
end

# this is the dispatcher for currently implemented delta functions organised by symbols
function simulate_markov_megpd(
    n_steps,
    kappa,
    sigma,
    xi,
    delta_name::Symbol;
    x0=1.0,
    burn_in_prop=0.0,
    show_progress::Bool=true
)

    delta = get_delta(delta_name)

    simulate_markov_megpd(
        n_steps,
        kappa,
        sigma,
        xi,
        delta;
        x0=x0,
        burn_in_prop=burn_in_prop,
        show_progress=show_progress
    )
end

function multiple_megpd_chains(
    n_chains::Int,
    n_steps::Int,
    kappa,
    sigma,
    xi,
    delta_name::Symbol;
    x0=1.0,
    burn_in_prop=0.0
)

    delta = get_delta(delta_name)

    multiple_megpd_chains(
        n_chains,
        n_steps,
        kappa,
        sigma,
        xi,
        delta;
        x0=x0,
        burn_in_prop=burn_in_prop
    )
end