# Joint MEGPD distribution with:
# R ~ EGPD(theta) where theta = (kappa, sigma, xi)
# Z ~ N(0, delta(R)^2)

function megpd_joint(
    x,
    y,
    theta,
    delta;
    lpdf = false
)

    if x <= 0.0 || y <= 0.0
        return lpdf ? -Inf : 0.0
    end

    r = x + y

    log_term_rad = egpd_lpdf(r, theta)

    if !isfinite(log_term_rad)
        return lpdf ? -Inf : 0.0
    end

    delta_r = max(delta(r), 0.01)

    log_ratio = log(x) - log(y)

    log_term_ang =
        -0.5 * log(2π) -
        log(delta_r) -
        0.5 * (log_ratio / delta_r)^2 +
        log(r) -
        log(x) -
        log(y)

    out = log_term_rad + log_term_ang

    return lpdf ? out : exp(out)
end

# Bernstein Dispatch
# R ~ EGPD(theta, w) where theta = (kappa, xi) and w are Bernstein weights

function megpd_joint(
    x,
    y,
    theta,
    delta,
    w;
    lpdf = false
)

    if x <= 0.0 || y <= 0.0
        return lpdf ? -Inf : 0.0
    end

    r = x + y

    log_term_rad = egpd_lpdf(r, theta, w)

    if !isfinite(log_term_rad)
        return lpdf ? -Inf : 0.0
    end

    delta_r = max(delta(r), 0.01)

    log_ratio = log(x) - log(y)

    log_term_ang =
        -0.5 * log(2π) -
        log(delta_r) -
        0.5 * (log_ratio / delta_r)^2 +
        log(r) -
        log(x) -
        log(y)

    out = log_term_rad + log_term_ang

    return lpdf ? out : exp(out)
end