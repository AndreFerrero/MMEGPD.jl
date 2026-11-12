# EGPD with CDF F(x) = H(x; sigma, xi)^kappa, where H is the GPD CDF.
# Everything below is built on the functions in gpd.jl.

function egpd_cdf(x, theta; lcdf = false)
  kappa = theta[1]
  sigma = theta[2]
  xi = theta[3]

  kappa = max(kappa, 1e-12)

  out = kappa * gpd_cdf(x, sigma, xi; lcdf = true)

  return lcdf ? out : exp(out)
end

function egpd_lpdf(x, theta; lpdf = true)
  kappa = theta[1]
  sigma = theta[2]
  xi = theta[3]
  # log of the GPD density; -Inf outside the support
  log_h = gpd_lpdf(x, sigma, xi; lpdf = true)
  isinf(log_h) && return lpdf ? -Inf : 0.0

  log_H = gpd_cdf(x, sigma, xi; lcdf = true)

  out = log(kappa) + log_h + (kappa - 1) * log_H

  return lpdf ? out : exp(out)
end

function egpd_quantile(p, theta)
  kappa = theta[1]
  sigma = theta[2]
  xi = theta[3]

  # F(x) = H(x)^kappa = p  <=>  x = H^{-1}(p^(1/kappa))
  return gpd_quantile(exp(log(p) / kappa), sigma, xi)
end

sample_egpd(theta) = egpd_quantile(rand(), theta)

sample_egpd(n::Integer, theta) = egpd_quantile.(rand(n), theta)


# Bernstein dispatch
# EGPD with cdf F(x) = B(H(x;sigma=1, xi)^kappa) where H is the GPD cdf
# Sigma is fixed to one to avoid identifiability problems with the Bernstein estimation of B
# Bernstein dependencies are in bernstein.jl

function egpd_cdf(x, theta, w; lcdf = false)
  kappa = theta[1]
  xi = theta[2]

  kappa = max(kappa, 1e-12)

  # exp(kappa * log H_xi) = H_xi ^ kappa
  uh = exp(kappa * gpd_cdf(x, 1.0, xi; lcdf = true))

  out = MMEGPD.bernstein_cdf(uh, w)
  
  return lcdf ? log(out) : out
end

function egpd_lpdf(x, theta, w; lpdf = true)

    kappa = max(theta[1], 1e-12)
    xi = theta[2]

    log_H = gpd_cdf(x, 1.0, xi; lcdf = true)

    log_h = gpd_lpdf(x, 1.0, xi; lpdf = true)

    isinf(log_h) && return lpdf ? -Inf : 0.0

    u = exp(kappa * log_H)

    log_b = log(MMEGPD.bernstein_density(u, w))

    out = log_b + log(kappa) +
          (kappa - 1) * log_H + log_h

    return lpdf ? out : exp(out)
end


function egpd_quantile(p, theta, w)

    kappa = max(theta[1], 1e-12)
    xi = theta[2]

    # Invert Bernstein CDF
    u = MMEGPD.bernstein_quantile(p, w)

    # Undo the EGPD power transformation
    h = exp(log(u) / kappa)

    # Invert the underlying GPD CDF (sigma fixed to 1)
    return gpd_quantile(h, 1.0, xi)
end

sample_egpd(theta, w) =
    egpd_quantile(rand(), theta, w)

sample_egpd(n::Integer, theta, w) =
    egpd_quantile.(rand(n), theta, w)
