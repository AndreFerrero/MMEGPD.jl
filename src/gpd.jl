"""
    log1mexp(a)

Numerically stable `log(1 - exp(a))` for `a <= 0`.
"""
@inline function log1mexp(a)
    a > 0 && return oftype(float(a), NaN)
    a == 0 && return oftype(float(a), -Inf)
    return a > -log(2) ? log(-expm1(a)) : log1p(-exp(a))
end

"""
    gpd_cdf(x, sigma, xi; lcdf = false)

CDF (or log-CDF if `lcdf = true`) of the GPD(sigma, xi).
"""
function gpd_cdf(x, sigma, xi; lcdf = false)
    sigma = max(sigma, 1e-12)

    x <= 0 && return lcdf ? -Inf : 0.0

    # log of the survival function, log(1 - G(x))
    if xi == 0.0
        log_surv = -x / sigma
    else
        t = 1 + xi * x / sigma
        # beyond the upper endpoint (xi < 0): CDF = 1
        t <= 0.0 && return lcdf ? 0.0 : 1.0
        log_surv = -log(t) / xi
    end

    out = log1mexp(log_surv)   # log G(x)

    # numerical protection
    out = max(out, log(1e-300))

    return lcdf ? out : exp(out)
end

"""
    gpd_lpdf(x, sigma, xi; lpdf = true)

Log-density (or density if `lpdf = false`) of the GPD(sigma, xi).
"""
function gpd_lpdf(x, sigma, xi; lpdf = true)
    out = _gpd_lpdf(x, sigma, xi)
    return lpdf ? out : exp(out)
end

function _gpd_lpdf(x, sigma, xi)
    x < 0 && return -Inf

    if xi == 0.0
        return -log(sigma) - x / sigma
    end

    t = 1 + xi * x / sigma
    t <= 0.0 && return -Inf

    return -log(sigma) + (-1 / xi - 1) * log(t)
end

"""
    gpd_quantile(p, sigma, xi)

Quantile function of the GPD(sigma, xi), for `p` in [0, 1].
"""
function gpd_quantile(p, sigma, xi)
    p <= 0 && return zero(float(p))
    if p >= 1
        return xi >= 0 ? Inf : -sigma / xi
    end

    if xi == 0.0
        return -sigma * log1p(-p)
    else
        return (sigma / xi) * expm1(-xi * log1p(-p))
    end
end

"""
    sample_gpd(sigma, xi)
    sample_gpd(n, sigma, xi)

Draw one (or `n`) samples from the GPD(sigma, xi).
"""
sample_gpd(sigma, xi) = gpd_quantile(rand(), sigma, xi)

sample_gpd(n::Integer, sigma, xi) = gpd_quantile.(rand(n), sigma, xi)

"""
    gpd_max_cdf(x, n, sigma, xi; lcdf = false)

CDF of the maximum of `n` i.i.d. GPD(sigma, xi) variables: G(x)^n.
"""
function gpd_max_cdf(x, n, sigma, xi; lcdf = false)
    log_F_max = n * gpd_cdf(x, sigma, xi; lcdf = true)
    return lcdf ? log_F_max : exp(log_F_max)
end