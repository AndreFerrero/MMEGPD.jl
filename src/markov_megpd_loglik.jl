# General markov likelihood logic

function _markov_megpd_loglik(
    x,
    basis,
    beta,
    theta,
    args...
)

    N = size(x, 1)

    ll = 0.0

    delta = delta_spline(basis, beta)

    #################################
    # Transition contribution
    #################################

    for t in 1:(N-1)

        ll += MMEGPD.megpd_joint(
            x[t, 1],
            x[t, 2],
            theta,
            delta,
            args...;
            lpdf=true
        )

    end

    #################################
    # Marginal correction
    #################################

    for t in 2:(N-1)

        ll -= MMEGPD.megpd_marginal(
            x[t, 1],
            theta,
            delta,
            args...;
            lpdf=true
        )

    end

    return ll
end

# Standard EGPD
function markov_megpd_loglik(
    x,
    basis,
    beta,
    theta
)
    return _markov_megpd_loglik(
        x, basis, beta, theta
    )
end


# Bernstein EGPD
function markov_megpd_loglik(
    x,
    basis,
    beta,
    theta,
    w
)
    return _markov_megpd_loglik(
        x, basis, beta, theta, w
    )
end

function _penalized_loglik(
    x,
    basis,
    S,
    beta,
    theta,
    lambda,
    args...
)

    ll = markov_megpd_loglik(
        x,
        basis,
        beta,
        theta,
        args...
    )

    penalty = 0.5 * lambda * beta' * S * beta

    return ll - penalty

end

# Standard EGPD
function penalized_loglik(
    x,
    basis,
    S,
    beta,
    theta,
    lambda
)

    return _penalized_loglik(
        x, basis, S, beta, theta, lambda
    )

end


# Bernstein EGPD
function penalized_loglik(
    x,
    basis,
    S,
    beta,
    theta,
    w,
    lambda
)

    return _penalized_loglik(
        x, basis, S, beta, theta, lambda, w
    )

end