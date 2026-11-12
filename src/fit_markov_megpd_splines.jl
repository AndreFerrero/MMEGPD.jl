# From unconstrained to original space
# ψ ----> η

function transform(
    psi;
    bernstein::Bool = false,
    m::Int = 5
)

    if !bernstein

        return vcat(
            exp(psi[1]),
            exp(psi[2]),
            psi[3:end]
        )

    else

        theta = [
            exp(psi[1]),
            psi[2]
        ]

        # m weights are represented by m-1 logits
        logits = psi[3:m+1]

        w = MMEGPD.softmax_weights(logits)

        beta = psi[m+2:end]

        return vcat(theta, w, beta)

    end

end

function objective(
    psi,
    x,
    basis,
    S,
    lambda,
    map::Bool = false;
    bernstein::Bool = false,
    m::Int = 5
)

    eta = transform(
        psi;
        bernstein = bernstein,
        m = m
    )

    if !bernstein

        # Standard EGPD
        theta = eta[1:3]
        beta = eta[4:end]

        obj = -MMEGPD.penalized_loglik(
            x,
            basis,
            S,
            beta,
            theta,
            lambda
        )

    else

        # Bernstein EGPD
        theta = eta[1:2]
        w = eta[3:m+2]
        beta = eta[m+3:end]

        obj = -MMEGPD.penalized_loglik(
            x,
            basis,
            S,
            beta,
            theta,
            w,
            lambda
        )

    end

    if map

        # Uniform priors on the positive parameters
        # transformed to logarithmic coordinates

        obj -= psi[1]

        if !bernstein
            obj -= psi[2]
        end

    end

    return obj

end

function fit_markov_megpd_splines(
  x, K::Int;
  # Splines settings
  spline=:crspline,
  knot_method=:quantile,
  # Cubic splines
  degree=3,
  # Penalisation matrix P-splines
  p=2,
  # Penalisation coefficient
  lambda=100.0,

  # Returning MAP estimates
  map::Bool=false,

  # Model specification
  bernstein::Bool=false,
  bernstein_m::Int=5,

  # Initial values
  beta_init=nothing,
  kappa_init=1.0,
  sigma_init=1.0,
  xi_init=0.1,
  w_init=nothing,

  # Optimisation settings
  verbose::Bool=true,
  show_every::Int=10,
  g_abstol=1e-8,
  x_reltol=0.0
)

    pairs = MMEGPD.as_pairs(x)

    R = pairs[:, 1] .+ pairs[:, 2]

    #################################
    # Model specification
    #################################

    len_theta = bernstein ? 2 : 3

    m = bernstein ? bernstein_m : 0

    sigma_model = bernstein ? 1.0 : sigma_init

    #################################
    # Bernstein initial weights
    #################################

    if bernstein

        if m < 1
            throw(ArgumentError(
                "bernstein_m must be at least 1."
            ))
        end

        if w_init === nothing

            w_init = fill(1.0 / m, m)

        else

            w_init = collect(w_init)

            if length(w_init) != m
                throw(ArgumentError(
                    "Length of w_init must equal bernstein_m."
                ))
            end

            if any(w_init .<= 0) || !all(isfinite, w_init)
                throw(ArgumentError(
                    "Initial Bernstein weights must be positive and finite."
                ))
            end

            w_init ./= sum(w_init)

        end

    end

    #################################
    # Radial support check
    #################################

    if xi_init < 0.0

        endpoint_init = -sigma_model / xi_init

    else

        endpoint_init = Inf

    end

    r_max = maximum(R)

    if r_max >= endpoint_init

        throw(ArgumentError(
            "Invalid initial parameters: the maximum observed " *
            "radius ($r_max) must be below the radial endpoint " *
            "($endpoint_init). For ξ=$xi_init and σ=$sigma_model, " *
            "the initial parameters do not provide valid support."
        ))

    end

    #################################
    # Spline construction
    #################################

    if spline == :pspline

        basis, = MMEGPD.bspline_basis(
            R;
            nbreaks=K,
            degree=degree
        )

        spline_coefs = length(basis)

        D = MMEGPD.difference_matrix(
            spline_coefs,
            p
        )

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

    #################################
    # Initial spline coefficients
    #################################

    if beta_init === nothing

        beta_init = zeros(spline_coefs)

    end

    #################################
    # Initial unconstrained vector
    #################################

    if !bernstein

        psi0 = vcat(
            log(kappa_init),
            log(sigma_init),
            xi_init,
            beta_init
        )

    else

        initial_logits = MMEGPD.weights_to_logits(w_init)

        psi0 = vcat(
            log(kappa_init),
            xi_init,
            initial_logits,
            beta_init
        )

    end

    #################################
    # Objective closure
    #################################

    negloglik = psi -> MMEGPD.objective(
        psi,
        pairs,
        basis,
        S,
        lambda,
        map;
        bernstein=bernstein,
        m=bernstein ? m : 5
    )

    #################################
    # Optimization
    #################################

    verbose && println("\nStarting optimisation...")

    result = Optim.optimize(
        negloglik,
        psi0,
        Optim.LBFGS(),
        Optim.Options(
            show_trace=verbose,
            extended_trace = true,
            show_every=show_every,
            g_abstol=g_abstol,
            x_reltol=x_reltol
        );
        autodiff=AutoForwardDiff()
    )

    psi_hat = Optim.minimizer(result)

    #################################
    # Natural parameter estimates
    #################################

    eta_hat = MMEGPD.transform(
        psi_hat;
        bernstein=bernstein,
        m=bernstein ? m : 5
    )

    if !bernstein

        theta_hat = eta_hat[1:3]

        beta_hat = eta_hat[4:end]

        w_hat = nothing

    else

        theta_hat = eta_hat[1:2]

        w_hat = eta_hat[3:m+2]

        beta_hat = eta_hat[m+3:end]

    end

    #################################
    # Penalised log likelihood
    #################################

    pen_loglik = -negloglik(psi_hat)

    #################################
    # Hessian on unconstrained scale
    #################################

    H = ForwardDiff.hessian(
        negloglik,
        psi_hat
    )

    H = Symmetric((H + H') / 2)

    V_psi = inv(H)

    se_psi = sqrt.(
        max.(diag(V_psi), 0.0)
    )

    #################################
    # Delta method
    #################################

    J = ForwardDiff.jacobian(
        psi -> MMEGPD.transform(
            psi;
            bernstein=bernstein,
            m=bernstein ? m : 5
        ),
        psi_hat
    )

    V_eta = J * V_psi * J'

    V_eta = Symmetric((V_eta + V_eta') / 2)

    se_eta = sqrt.(
        max.(diag(V_eta), 0.0)
    )

    #################################
    # Extract standard errors
    #################################

    se_theta = se_eta[1:len_theta]

    if !bernstein

        se_kappa = se_eta[1]
        se_sigma = se_eta[2]
        se_xi = se_eta[3]

        se_w = nothing

    else

        se_kappa = se_eta[1]
        se_sigma = nothing
        se_xi = se_eta[2]

        se_w = se_eta[3:m+2]

    end

    se_beta = se_eta[end-length(beta_hat)+1:end]

    #################################
    # Return
    #################################

    return (

        result=result,

        basis=basis,
        pen_loglik=pen_loglik,
        lambda=lambda,

        # Model specification
        bernstein=bernstein,
        len_theta=len_theta,
        bernstein_m=bernstein ? m : nothing,

        # Optimization-scale quantities
        psi=psi_hat,
        hessian=H,
        vcov_psi=V_psi,
        se_psi=se_psi,

        # Natural-scale quantities
        eta=eta_hat,
        theta=Tuple(theta_hat),
        beta=beta_hat,
        w=w_hat,

        vcov=V_eta,
        se=se_eta,

        # Individual standard errors
        se_theta=se_theta,
        se_kappa=se_kappa,
        se_sigma=se_sigma,
        se_xi=se_xi,
        se_w=se_w,
        se_beta=se_beta,

        # Data information
        x=x,
        R=R,
        map=map,
        S=S,
        K=K,
        p=p,
        degree=degree
    )

end