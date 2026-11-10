# EGPD with CDF F(x) = G(x; sigma, xi)^kappa, where G is the GPD CDF.
# Everything below is built on the functions in gpd.jl.

function egpd_cdf(x, kappa, sigma, xi; lcdf = false)
    kappa = max(kappa, 1e-12)

    out = kappa * gpd_cdf(x, sigma, xi; lcdf = true)

    return lcdf ? out : exp(out)
end

function egpd_lpdf(x, kappa, sigma, xi; lpdf = true)
    # log of the GPD density; -Inf outside the support
    log_g = gpd_lpdf(x, sigma, xi; lpdf = true)
    isinf(log_g) && return lpdf ? -Inf : 0.0

    log_G = gpd_cdf(x, sigma, xi; lcdf = true)

    out = log(kappa) + log_g + (kappa - 1) * log_G

    return lpdf ? out : exp(out)
end

function egpd_quantile(p, kappa, sigma, xi)
    # F(x) = G(x)^kappa = p  <=>  x = G^{-1}(p^(1/kappa))
    return gpd_quantile(exp(log(p) / kappa), sigma, xi)
end

sample_egpd(kappa, sigma, xi) = egpd_quantile(rand(), kappa, sigma, xi)

sample_egpd(n::Integer, kappa, sigma, xi) = egpd_quantile.(rand(n), kappa, sigma, xi)