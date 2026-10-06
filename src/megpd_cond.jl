function megpd_cond(
    x_next,
    x_prev,
    kappa,
    sigma,
    xi,
    delta;
    lpdf = false
)

    # --------------------------------------------------------
    # Joint density / log-density
    # --------------------------------------------------------

    joint =
        MMEGPD.megpd_joint(
            x_prev,
            x_next,
            kappa,
            sigma,
            xi,
            delta;
            lpdf = lpdf
        )

    # --------------------------------------------------------
    # Handle invalid / zero joint
    # --------------------------------------------------------

    if !isfinite(joint)

        return lpdf ? -Inf : 0.0

    end

    # --------------------------------------------------------
    # Normalising marginal density / log-density
    #
    # megpd_marginal returns:
    #   lpdf = false -> marginal density
    #   lpdf = true  -> log marginal density
    # --------------------------------------------------------

    marg =
        MEGPD.megpd_marginal(
            x_prev,
            kappa,
            sigma,
            xi,
            delta;
            lpdf = lpdf
        )

    # --------------------------------------------------------
    # Invalid / zero marginal
    # --------------------------------------------------------

    if lpdf

        if !isfinite(marg)

            return -Inf

        end

        # log p(y|x) = log g(x,y) - log g(x)
        return joint - marg

    else

        if !isfinite(marg) || marg <= 0.0

            return 0.0

        end

        # p(y|x) = g(x,y) / g(x)
        return joint / marg

    end

end