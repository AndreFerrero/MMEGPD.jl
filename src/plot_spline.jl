function spline_confidence_band(
    fit,
    r_grid;
    extrapolate = true,
    level = 0.95
)

    basis = fit.basis
    beta = fit.beta

    # Covariance matrix of beta
    #
    # fit.vcov is on the natural parameter scale:
    # [kappa, sigma, xi, beta...]
    V_beta = fit.vcov[4:end, 4:end]

    # Normal critical value
    z = quantile(
        Normal(),
        1 - (1 - level) / 2
    )

    # Allow missing values when extrapolate = false
    delta_hat = Vector{Union{Missing, Float64}}(undef, length(r_grid))
    se_logdelta = Vector{Union{Missing, Float64}}(undef, length(r_grid))

    lower = Vector{Float64}(undef, length(r_grid))
    upper = Vector{Float64}(undef, length(r_grid))

    r_max = basis.breakpoints[end]

    for (i, r) in enumerate(r_grid)

        if r > r_max

            if extrapolate

                # Extrapolate by keeping the function constant
                # at the endpoint
                r_eval = r_max

            else

                # No extrapolation
                delta_hat[i] = missing
                se_logdelta[i] = missing
                lower[i] = NaN
                upper[i] = NaN

                continue
            end

        else

            # Within the spline domain
            r_eval = r

        end

        B = MMEGPD.basis_matrix(
            basis,
            [r_eval]
        )

        b = B[1, :]

        # Estimated log delta
        logdelta_hat = dot(b, beta)

        # Estimated delta
        delta_hat[i] = exp(logdelta_hat)

        # SE on log(delta) scale
        se_logdelta[i] = sqrt(
            max(
                dot(b, V_beta * b),
                0.0
            )
        )

        # CI on log scale, transformed back
        lower[i] = exp(
            logdelta_hat - z * se_logdelta[i]
        )

        upper[i] = exp(
            logdelta_hat + z * se_logdelta[i]
        )
    end

    return (
        estimate = delta_hat,
        se_logdelta = se_logdelta,
        lower = lower,
        upper = upper
    )
end

function plot_spline(
    fit;
    delta_true = nothing,
    plot_fn = true,
    level = 0.95,
    show_ci = true,
    extrapolate = true
)
    R = fit.R
    r_grid = collect(range(
        minimum(R),
        maximum(R),
        length=500
    ))

    # Estimated spline
    spline_ci = spline_confidence_band(
        fit,
        r_grid;
        extrapolate=extrapolate,
        level = level
    )

    delta_hat = spline_ci.estimate
    lower = spline_ci.lower
    upper = spline_ci.upper

    # True delta, only if supplied
    if delta_true !== nothing
        delta_true_fn = MMEGPD.get_delta(delta_true)
        delta_true_vals = [
            delta_true_fn(r) for r in r_grid
        ]
    end

    if plot_fn

        if delta_true !== nothing

            p = plot(
                r_grid,
                delta_true_vals,
                linewidth = 3,
                color = :black,
                label = "true δ(r)",
                xlabel = "r",
                ylabel = "δ(r)"
            )

        else

            p = plot(
                r_grid,
                delta_hat,
                linewidth = 3,
                color = :blue,
                label = "estimated δ(r)",
                xlabel = "r",
                ylabel = "δ(r)"
            )

        end

        # Confidence band
        if show_ci

            plot!(
                p,
                r_grid,
                lower,
                fillrange = upper,
                fillalpha = 0.2,
                linealpha = 0,
                color = :blue,
                label = "$(round(Int, 100*level))% CI"
            )

        end

        # Estimated spline
        plot!(
            p,
            r_grid,
            delta_hat,
            linewidth = 3,
            color = :blue,
            label = "estimated δ(r)"
        )

        return p
    end

    return (
        estimate = delta_hat,
        lower = lower,
        upper = upper,
        se_logdelta = spline_ci.se_logdelta
    )
end