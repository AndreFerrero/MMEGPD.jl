function megpd_marginal(
    x,
    kappa,
    sigma,
    xi,
    delta;
    lpdf = false,
    logR = false,
    logRshift = 0.0,
    hyper_inv_sin = false
)

    marg, err =
        quadgk(
            y ->
                megpd_joint(
                    x,
                    y,
                    kappa,
                    sigma,
                    xi,
                    delta;
                    logR = logR,
                    logRshift = logRshift,
                    hyper_inv_sin = hyper_inv_sin
                ),
            1e-10,
            Inf
        )

    # --------------------------------------------------------
    # Handle invalid / zero marginal
    # --------------------------------------------------------

    if !isfinite(marg) || marg <= 0.0

        return lpdf ? -Inf : 0.0

    end

    return lpdf ? log(marg) : marg
end