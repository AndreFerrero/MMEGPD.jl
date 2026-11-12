# Bernstein dependencies for the construction of the Bernstein EGPD

"""
    bernstein_basis(u, k, m)

Bernstein basis function of order `m`: the Beta(k, m-k+1) density at `u`,
i.e. m * binom(m-1, k-1) * u^(k-1) * (1-u)^(m-k).
"""
function bernstein_basis(u, k, m)
    (u < 0 || u > 1) && return zero(float(u))
    return m * binomial(m - 1, k - 1) * u^(k - 1) * (1 - u)^(m - k)
end

"""
    bernstein_density(u, w)

Bernstein density b(u) = sum_k w[k] * bernstein_basis(u, k, m), with m = length(w).
Returned on the natural scale.
"""
function bernstein_density(u, w)
    m = length(w)
    return sum(w[k] * bernstein_basis(u, k, m) for k in 1:m)
end

"""
    bernstein_cdf(u, w)

CDF B(u) of the Bernstein density, on the natural scale. Uses
B(u) = sum_{j=1}^m W_j * binom(m, j) * u^j * (1-u)^(m-j),  W = cumsum(w).
"""
function bernstein_cdf(u, w)
    u <= 0 && return 0.0
    u >= 1 && return 1.0

    m = length(w)
    W = cumsum(w)
    val = sum(W[j] * binomial(m, j) * u^j * (1 - u)^(m - j) for j in 1:m)

    return clamp(val, 0.0, 1.0)
end

"""
    bernstein_quantile(p, w)

Quantile function of the Bernstein distribution defined by weights w.
"""
function bernstein_quantile(p, w)

    if !(0.0 <= p <= 1.0)
        throw(ArgumentError("p must be between 0 and 1."))
    end

    p == 0.0 && return 0.0
    p == 1.0 && return 1.0

    return find_zero(
        u -> MMEGPD.bernstein_cdf(u, w) - p,
        (0.0, 1.0),
        Bisection()
    )
end

"""
    softmax_weights(w)

Weights w = softmax([0; w]), of length `length(w) + 1`.
"""
function softmax_weights(w)
    z = vcat(zero(eltype(w)), w)
    z = z .- maximum(z)
    e = exp.(z)
    return e ./ sum(e)
end

"""
    weights_to_logits(softw)

Inverse of `softmax_weights`: w = log(softw[2:end] / softw[1]), with a small floor
so that zero weights give finite starting values.
"""
function weights_to_logits(softw)
    floor = 1e-8
    return log.(max.(softw[2:end], floor) ./ max(softw[1], floor))
end