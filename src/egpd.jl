function egpd_cdf(x, kappa, sigma, xi; lcdf = false)
    sigma = max(sigma, 1e-12)
    kappa = max(kappa, 1e-12)

    if xi == 0.0
        gpd_cdf = 1 - exp(-x/sigma)
    else
        t = 1 + xi*x/sigma
        t <= 0.0 && return lcdf ? -Inf : 0.0

        gpd_cdf = 1 - exp((-1/xi) * log(t))
    end

    # numerical protection
    gpd_cdf = clamp(gpd_cdf, 1e-300, 1.0)

    out = kappa * log(gpd_cdf)

    return lcdf ? out : exp(out)
end

function egpd_lpdf(x, kappa, sigma, xi; lpdf = true)
    if xi == 0.0
        return log(kappa) - log(sigma) - x/sigma + (kappa-1)*log(1 - exp(-x/sigma))
    end
    t = 1 + xi*x/sigma
    t <= 0.0 && return -Inf

    # log pdf of the GPD 
    gpd_pdf = -log(sigma) + (-1/xi - 1)*log(t)

    gpd_cdf = max(1 - exp(-1/xi * log(t)), 1e-300)

    out = log(kappa) + gpd_pdf + (kappa-1)*log(gpd_cdf)

    return lpdf ? out : exp(out)
end

function egpd_quantile(p, kappa, sigma, xi)

    if xi == 0.0
        return -sigma * log1p(-p^(1 / kappa))
    else
        return (sigma / xi) *
               ((1 - p^(1 / kappa))^(-xi) - 1)
    end

end

function sample_egpd(
    kappa,
    sigma,
    xi
)

    u = rand()

    return egpd_quantile(
        u,
        kappa,
        sigma,
        xi
    )

end

function sample_egpd(
    n::Integer,
    kappa,
    sigma,
    xi
)

    u = rand(n)

    return egpd_quantile.(
        u,
        kappa,
        sigma,
        xi
    )

end

function egpd_max_cdf(x, n, kappa, sigma, xi; lcdf = false)

    # CDF of a single EGPD observation
    log_F = egpd_cdf(
        x,
        kappa,
        sigma,
        xi;
        lcdf = true
    )

    # CDF of the maximum:
    #
    # P(M_n <= x) = P(X_1 <= x, ..., X_n <= x)
    #              = F(x)^n
    #
    log_F_max = n * log_F

    return lcdf ? log_F_max : exp(log_F_max)
end
