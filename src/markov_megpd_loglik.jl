function markov_megpd_loglik(
    x,
    basis,
    beta,
    kappa,
    sigma,
    xi,
    logR,
    logRshift
)
    # x is assumed to be the pairs matrix

    N = size(x, 1)

    ll = 0.0


    # create delta function
    delta =
    delta_spline(
        basis,
        beta
    )


    #################################
    # transition contribution
    #################################

    for t in 1:N-1

        ll +=
        MMEGPD.megpd_joint(
            x[t, 1],
            x[t, 2],
            kappa,
            sigma,
            xi,
            delta;
            lpdf=true,
            logR = logR,
            logRshift = logRshift
        )

    end


    #################################
    # marginal correction
    #################################

    for t in 2:N-1

        ll -= MMEGPD.megpd_marginal(
            x[t, 1],
            kappa,
            sigma,
            xi,
            delta;
            lpdf=true,
            logR = logR,
            logRshift = logRshift
        )

    end


    return ll

end

function penalized_loglik(
    x,
    basis,
    S,
    beta,
    kappa,
    sigma,
    xi,
    lambda,
    logR,
    logRshift
)

    ll =
    markov_megpd_loglik(
        x,
        basis,
        beta,
        kappa,
        sigma,
        xi,
        logR,
        logRshift
    )


    penalty =
    0.5 *
    lambda *
    beta' *
    S *
    beta


    return ll - penalty

end