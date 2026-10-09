# -----------------------------------------------------------------------
# 1. Penalty matrix utilities
# -----------------------------------------------------------------------

"""
    pseudo_logdet(P; tol=1e-8)

Return (r, logdetP_plus): the numerical rank of P and the log of the
pseudo-determinant (product of the r nonzero eigenvalues), using a
threshold relative to the largest eigenvalue rather than an exact-zero
test (robust to floating point noise in the theoretically-zero
eigenvalues of the difference penalty's null space).
"""
function pseudo_logdet(P::AbstractMatrix; tol::Float64=1e-8)
    ev = real.(eigvals(Symmetric(Matrix(P))))
    thresh = tol * maximum(ev)
    nz = ev[ev .> thresh]
    r = length(nz)
    logdetP = sum(log.(nz))
    return r, logdetP
end

# -----------------------------------------------------------------------
# 2. Laplace approximation to log p(x | lambda) at a single MAP fit
# -----------------------------------------------------------------------

"""
    laplace_log_marginal(fit; tol_rank=1e-8, drop_constant=true)

Laplace-approximate log p(x | lambda) evaluated at the MAP returned by
`fit_markov_megpd_splines` (via the patched contract above).

    log p(x|lambda) ~= -negloglik(psi) + (r/2) log(lambda)
                       - (1/2) logdet(H_lambda)  [+ constants]

`psi` is the UNCONSTRAINED vector [log kappa, log sigma, xi, beta]
-- the same space the objective was actually optimised in. H_lambda is
the Hessian of the penalised NEGATIVE log-likelihood (= observed
information) in that space, computed via ForwardDiff on the *same*
`negloglik` closure that was optimised, and its log-determinant is
obtained via Cholesky (never via `det`) for numerical stability.

The penalty matrix used for the pseudo-determinant/rank is `fit.S`
exactly as returned by the fit (not rebuilt from K,p), guaranteeing
consistency with what was actually optimised.

If `drop_constant = true` (default), lambda-independent terms
(pseudo-determinant of S, dimension-counting log(2*pi) terms) are
omitted -- fine for finding argmax(lambda), NOT fine if you want the
actual evidence value / Bayes factors across models.
"""
function laplace_log_marginal(fit;
    tol_rank::Float64=1e-8,
    drop_constant::Bool=false
)

    # Extract relevant objects
    psi = fit.psi
    pen_loglik_at_map = fit.pen_loglik

    S, lambda = fit.S, fit.lambda

    d = length(psi)

    H = fit.hessian

    # Compute hessian log determinant via Cholesky decomposition
    local logdetH
    try
        C = cholesky(H)
        logdetH = 2 * sum(log.(diag(C.U)))
    catch e
        @warn "Hessian not positive definite at lambda=$(lambda); returning -Inf" exception = e
        return -Inf
    end

    r, logdetS = pseudo_logdet(S; tol=tol_rank)

    marg_loglik = pen_loglik_at_map + (r / 2) * log(lambda) - 0.5 * logdetH

    if !drop_constant
        marg_loglik += 0.5 * logdetS + 0.5 * (d - r) * log(2*pi)
    end

    return marg_loglik
end

function numerical_log_marginal(
    fit;
    x,
    se_w = 1.0,
    rtol = 1e-3,
    atol = 1e-8,
    maxevals = 100_000,
    tol_rank = 1e-8,
    drop_constant = false
)

    psi_hat = fit.psi
    se_psi = fit.se_psi
    S = fit.S
    lambda = fit.lambda
    d = length(psi_hat)

    pairs = MMEGPD.as_pairs(x)

    # --------------------------------------------------------
    # Log of the integrand at the mode
    # --------------------------------------------------------

    log_mode = -MMEGPD.objective(
        psi_hat,
        pairs,
        fit.basis,
        S,
        lambda;
        map=true
    )


    lower = psi_hat .- se_w .* se_psi
    upper = psi_hat .+ se_w .* se_psi

    # --------------------------------------------------------
    # Stabilised integrand
    # --------------------------------------------------------

    integrand(psi) = begin

        z = -MMEGPD.objective(
            psi,
            pairs,
            fit.basis,
            S,
            lambda;
            map=true
        )

        isfinite(z) ? exp(z - log_mode) : 0.0
    end

    val, err = hcubature(
        integrand,
        lower,
        upper;
        rtol=rtol,
        atol=atol,
        maxevals=maxevals
    )

    if !isfinite(val) || val <= 0.0
        return -Inf, err
    end

    # --------------------------------------------------------
    # Numerical log integral
    # --------------------------------------------------------

    log_marg_lik =
        log_mode + log(val)

    # --------------------------------------------------------
    # Same constants as Laplace approximation
    # --------------------------------------------------------

    r, logdetS = MMEGPD.pseudo_logdet(
        S;
        tol=tol_rank
    )

    # lambda-dependent normalisation
    log_marg_lik += (r / 2) * log(lambda)

    # lambda-independent constants
    if !drop_constant
        log_marg_lik +=
            0.5 * logdetS +
            0.5 * (d - r) * log(2π)
    end

    return log_marg_lik, err
end

# -----------------------------------------------------------------------
#  Continuous optimisation over log10(lambda)
# -----------------------------------------------------------------------

"""
    laplace_optimize(x, K; degree=3, p=1, lower=-4.0, upper=8.0,
                      beta_init=nothing, kappa_init=1.0, sigma_init=1.0,
                      xi_init=0.1, tol_rank=1e-8)

Maximise the Laplace-approximate log marginal likelihood over
log10(lambda) using Brent's method (derivative-free 1-D optimisation
over a bounded interval -- avoids differentiating through the inner
penalised-MLE argmax via the envelope theorem, which is not implemented
here). Refits the penalised MLE at every proposed lambda, warm-started
from the previous evaluation.
"""
function laplace_optimize(x, K::Int64;
    spline=:pspline, knot_method=:even,
    degree::Int=3, p::Int=2,
    map=true,
    lower::Float64=0.0, upper::Float64=4.0,
    tol_rank::Float64=1e-8,
    beta_init=nothing, kappa_init::Float64=1.0,
    sigma_init::Float64=1.0, xi_init::Float64=0.1,
    verbose::Bool=false,
    x_reltol::Float64=0.0, g_abstol::Float64=1e-8
)

    # mutable warm-start cache closed over by the objective
    cache = Ref{Any}((beta=beta_init, kappa=kappa_init, sigma=sigma_init, xi=xi_init))

    function marg_loglik(log10_lam::Float64)
        lam = 10.0 ^ log10_lam
        c = cache[]
        fit = MMEGPD.fit_markov_megpd_splines(
            x, K; spline=spline, knot_method=knot_method,
            degree=degree, p=p, lambda=lam,
            map=map,
            beta_init=c.beta, kappa_init=c.kappa,
            sigma_init=c.sigma, xi_init=c.xi,
            verbose=verbose, x_reltol=x_reltol, g_abstol=g_abstol)
        cache[] = (beta=fit.beta, kappa=fit.theta[1],
            sigma=fit.theta[2], xi=fit.theta[3])
        val = laplace_log_marginal(fit; tol_rank=tol_rank, drop_constant=true)
        return isfinite(val) ? -val : Inf   # Optim minimises
    end

    res = Optim.optimize(marg_loglik, lower, upper, Brent())

    best_log10_lambda = Optim.minimizer(res)
    best_lambda = 10.0 ^ best_log10_lambda
    best_logL = -Optim.minimum(res)

    c = cache[]
    best_fit = MMEGPD.fit_markov_megpd_splines(
        x, K; spline=spline, knot_method=knot_method,
        degree=degree, p=p,
        lambda=best_lambda,
        map=false,
        beta_init=c.beta, kappa_init=c.kappa,
        sigma_init=c.sigma, xi_init=c.xi,
        verbose=verbose, x_reltol=x_reltol)

    return (best_lambda=best_lambda, best_log10_lambda=best_log10_lambda,
        best_logL=best_logL, best_fit=best_fit, optim_result=res)
end



function laplace_grid_search(
    x, K::Int64;
    spline=:pspline,
    knot_method=:even,
    degree::Int=3,
    p::Int=2,
    map::Bool=true,
    lower::Float64=0.0,
    upper::Float64=3.0,
    ngrid::Int=50,
    beta_init=nothing,
    kappa_init::Float64=1.0,
    sigma_init::Float64=1.0,
    xi_init::Float64=0.1,
    tol_rank::Float64=1e-8,
    verbose::Bool=false,
    x_reltol::Float64=0.0,
    g_abstol::Float64=1e-8
)

    # Grid in log10(lambda)
    log10_lambda_grid = collect(range(lower, upper; length=ngrid))
    lambda_grid = 10.0 .^ log10_lambda_grid

    # Storage
    logL_grid = fill(NaN, ngrid)

    # Warm-start cache
    cache = Ref{Any}((
        beta=beta_init,
        kappa=kappa_init,
        sigma=sigma_init,
        xi=xi_init
    ))

    for i in eachindex(lambda_grid)
        println("Evaluating lambda = $(lambda_grid[i]), remaining = $(ngrid - i)")
        λ = lambda_grid[i]

        c = cache[]

        try
            fit = MMEGPD.fit_markov_megpd_splines(
                x, K;
                spline=spline,
                knot_method=knot_method,
                degree=degree,
                p=p,
                lambda=λ,
                map=map,
                beta_init=c.beta,
                kappa_init=c.kappa,
                sigma_init=c.sigma,
                xi_init=c.xi,
                verbose=verbose,
                x_reltol=x_reltol,
                g_abstol=g_abstol
            )

            # Update warm start
            cache[] = (
                beta=fit.beta,
                kappa=fit.theta[1],
                sigma=fit.theta[2],
                xi=fit.theta[3]
            )

            # Evaluate Laplace approximation
            val = laplace_log_marginal(
                fit;
                tol_rank=tol_rank,
                drop_constant=true
            )

            logL_grid[i] = val

        catch err
            @warn "Failed for lambda = $λ" exception=err
        end
    end

    return (
        lambda=lambda_grid,
        log10lambda=log10_lambda_grid,
        logL=logL_grid
    )
end