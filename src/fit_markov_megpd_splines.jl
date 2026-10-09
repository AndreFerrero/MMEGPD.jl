# From unconstrained to natural space
# ψ ----> η

function transform(psi)
    return vcat(
        exp(psi[1]),
        exp(psi[2]),
        psi[3:end]
    )
end

function objective(
    psi,
    x, # Matrix of pairs
    basis,
    S,
    lambda;
    map::Bool=false,
    logR::Bool=false,
    logRshift=0
)
    eta = transform(psi)

    # Transform parameters
    kappa = eta[1]
    sigma = eta[2]
    xi = eta[3]
    beta = eta[4:end]

    # Negative penalised log likelihood
    obj = -MMEGPD.penalized_loglik(
        x,
        basis,
        S,
        beta,
        kappa,
        sigma,
        xi,
        lambda,
        logR,
        logRshift
    )

    if map
        # Add terms required by the chosen prior specification
        #
        # Example: uniform priors on kappa and sigma
        # when optimising in log(kappa), log(sigma) coordinates.
        obj -= psi[1]
        obj -= psi[2]
    end

    return obj
end

function fit_markov_megpd_splines(
    x, K::Int64; spline=:crspline, knot_method=:even,
    logR::Bool=false, logRshift=0.0,
    degree=3, p=2, lambda=100.0, map::Bool=false,
    beta_init=nothing, kappa_init=1.0, sigma_init=1.0, xi_init=0.1,
    verbose::Bool=true, show_every::Int=10, g_abstol=1e-8, x_reltol=0.0
)
    # the given x can be both the series or the pairs matrix already
    pairs = MMEGPD.as_pairs(x)

    R = pairs[:, 1] .+ pairs[:, 2]

    if logR==true
        R = log.(R .+ logRshift)
    end

    if spline == :pspline

        basis, = MMEGPD.bspline_basis(
            R;
            nbreaks=K,
            degree=degree
        )

        spline_coefs = length(basis)

        D = MMEGPD.difference_matrix(spline_coefs, p)
        S = D' * D

    elseif spline == :crspline

        basis = MMEGPD.crspline_basis(
            R;
            nbreaks=K,
            knot_method=knot_method
        )

        S = basis.S

        spline_coefs = K
    else

        throw(ArgumentError(
            "Unknown spline type: $spline"
        ))

    end

    if beta_init === nothing
        beta_init = zeros(spline_coefs)
    end

    psi0 = vcat(log(kappa_init), log(sigma_init), xi_init, beta_init)

    # the exact objective closure -- reused for both optimisation and
    # the Hessian, so they can never silently drift apart
    negloglik = psi -> MMEGPD.objective(psi, pairs, basis, S, lambda; map=map, logR = logR, logRshift = logRshift)

    verbose && println("\n Starting optimisation...")

    result = Optim.optimize(
        negloglik, psi0, Optim.LBFGS(),
        Optim.Options(
            show_trace=verbose, show_every=show_every, g_abstol=g_abstol, x_reltol=x_reltol);
        autodiff=AutoForwardDiff()
    )

    psi_hat = Optim.minimizer(result)      # UNCONSTRAINED vector:
    # [log kappa, log sigma, xi, beta]

    eta_hat = transform(psi_hat)

    # Optimum value of the penalised log likelihood
    pen_loglik = - negloglik(psi_hat)

    # Hessian on the optimisation (unconstrained) scale
    H = ForwardDiff.hessian(negloglik, psi_hat)

    # Numerical symmetry safeguard
    H = Symmetric((H + H') / 2)

    # Covariance matrix on the unconstrained psi scale
    V_psi = inv(H)
    se_psi = sqrt.(max.(diag(V_psi), 0.0))

    # Delta-method transformation to natural parameter scale
    # Jacobian d(eta) / d(psi)
    J = ForwardDiff.jacobian(transform, psi_hat)

    # Delta-method covariance matrix on natural scale
    V_eta = J * V_psi * J'

    # Numerical symmetry safeguard
    V_eta = Symmetric((V_eta + V_eta') / 2)

    # Standard errors
    se_eta = sqrt.(max.(diag(V_eta), 0.0))

    kappa_hat = eta_hat[1]
    sigma_hat = eta_hat[2]
    xi_hat = eta_hat[3]
    beta_hat = eta_hat[4:end]

    return (
        result=result,
        basis=basis, pen_loglik,
        lambda=lambda,

        # Optimisation-scale quantities
        psi=psi_hat,
        hessian=H,
        vcov_psi=V_psi,
        se_psi = se_psi,
        
        # Natural-scale quantities
        eta=eta_hat,
        theta=(kappa_hat, sigma_hat, xi_hat),
        beta=beta_hat,
        vcov=V_eta,
        se=se_eta,

        # Individual standard errors
        se_kappa=se_eta[1],
        se_sigma=se_eta[2],
        se_xi=se_eta[3],
        se_beta=se_eta[4:end],
        
        # data information
        x=x,
        R = R,
        S=S, K=K,
        p=p,
        degree=degree,
        logR=logR,
        logRshift=logRshift
    )
end