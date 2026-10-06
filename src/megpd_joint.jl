function megpd_joint(
    x,
    y,
    kappa,
    sigma,
    xi,
    delta;
    lpdf = false
)

    # Support is x,y > 0
    if x <= 0.0 || y <= 0.0
        return 0.0
    end

    r = x + y

    if r <= 0.0
        return 0.0
    end

    # Radial component
    log_term_rad = egpd_lpdf(r, kappa, sigma, xi)

    if !isfinite(log_term_rad)
        return lpdf ? -Inf : 0.0
    end

    # Angular scale
    delta_r = max(delta(r), 0.01)

    # log(x/y)
    log_ratio = log(x) - log(y)

    # Log Gaussian density
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