# Marginal integration of the Joint MEGPD

function megpd_marginal(
    x,
    theta,
    delta;
    lpdf = false
)

    marg, err =
        quadgk(
            y ->
                megpd_joint(
                    x,
                    y,
                    theta,
                    delta
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

# Bernstein dispatch

function megpd_marginal(
    x,
    theta,
    w,
    delta;
    lpdf = false
)

    marg, err =
        quadgk(
            y ->
                megpd_joint(
                    x,
                    y,
                    theta,
                    w,
                    delta
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