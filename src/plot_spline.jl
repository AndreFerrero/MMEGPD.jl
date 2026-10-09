function spline_confidence_band(
    fit,
    r_grid;
    extrapolate = true,
    level = 0.95
)
    # Indicator of the scale used for estimation
    r_logscale = fit.logR

    basis = fit.basis
    beta = fit.beta

    # Covariance matrix of beta
    #
    # fit.vcov is on the natural parameter scale:
    # [kappa, sigma, xi, beta...]
    V_beta = fit.vcov[4:end, 4:end]

    # Normal critical value
    zcrit = quantile(
        Normal(),
        1 - (1 - level) / 2
    )

    n = length(r_grid)

    # These may contain missing values when extrapolate = false
    delta_hat = Vector{Union{Missing, Float64}}(undef, n)
    se_logdelta = Vector{Union{Missing, Float64}}(undef, n)

    # Keep these as ordinary Float64 vectors.
    #
    # NaN is used outside the spline domain when
    # extrapolate = false. This is preferable to Missing
    # for plotting.
    lower = Vector{Float64}(undef, n)
    upper = Vector{Float64}(undef, n)

    z_min = basis.breakpoints[1]
    z_max = basis.breakpoints[end]

    for (i, r) in enumerate(r_grid)

        # Convert natural radius r into the coordinate used
        # by the spline basis.
        if r_logscale == true

            if r <= 0.0
                error(
                    "All values of r_grid must be positive when " *
                    "r_logscale == true"
                )
            end

            z_eval = log(r)

        else
            z_eval = r
        end

        # --------------------------------------------------------------
        # Check whether the requested point is inside the fitted
        # spline domain.
        # --------------------------------------------------------------

        if z_eval < z_min || z_eval > z_max

            if extrapolate

                # Constant endpoint extrapolation.
                #
                # Notice that we clamp in the SPLINE coordinate,
                # not in the natural radius coordinate.
                z_eval = clamp(
                    z_eval,
                    z_min,
                    z_max
                )

            else

                # Estimated quantities are allowed to be missing.
                delta_hat[i] = missing
                se_logdelta[i] = missing

                # CI vectors must remain Float64 for plotting.
                # NaN tells Plots.jl to leave a gap.
                lower[i] = NaN
                upper[i] = NaN

                continue
            end
        end

        # --------------------------------------------------------------
        # Evaluate the spline basis in its OWN coordinate.
        # --------------------------------------------------------------

        B = MMEGPD.basis_matrix(
            basis,
            [z_eval]
        )

        b = B[1, :]

        # Estimated log(delta)
        logdelta_hat = dot(b, beta)

        # Estimated delta
        delta_hat[i] = exp(logdelta_hat)

        # Standard error on log(delta) scale
        se_logdelta[i] = sqrt(
            max(
                dot(b, V_beta * b),
                0.0
            )
        )

        # Confidence interval on log(delta) scale,
        # transformed back to delta scale.
        lower[i] = exp(
            logdelta_hat -
            zcrit * se_logdelta[i]
        )

        upper[i] = exp(
            logdelta_hat +
            zcrit * se_logdelta[i]
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
    extrapolate = true,
    xscale = :natural,
    n_grid = 500
)

    # ------------------------------------------------------------------
    # fit.R is the coordinate used during fitting.
    #
    # If spline_scale == :log:
    #     fit.R = log(R)
    #
    # If spline_scale == :natural:
    #     fit.R = R
    # ------------------------------------------------------------------

    R_fit = fit.R

    # Recover the natural-radius values represented by fit.R.
    if fit.logR == true

        R_natural = exp.(R_fit)

    else

        R_natural = R_fit
    end
    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Always construct the evaluation grid in NATURAL R.
    #
    # The confidence-band function will transform r -> log(r)
    # if the spline was fitted on log(R).
    # ------------------------------------------------------------------

    r_grid = collect(range(
        minimum(R_natural),
        maximum(R_natural),
        length = n_grid
    ))

    # ------------------------------------------------------------------
    # Evaluate fitted spline
    # ------------------------------------------------------------------

    spline_ci = spline_confidence_band(
        fit,
        r_grid;
        extrapolate = extrapolate,
        level = level
    )

    delta_hat = spline_ci.estimate
    lower = spline_ci.lower
    upper = spline_ci.upper

    # ------------------------------------------------------------------
    # True delta
    #
    # delta_true is assumed to be defined on the NATURAL radius scale:
    #
    #     delta_true_fn(r)
    #
    # Therefore we ALWAYS evaluate it using r_grid, never fit.R.
    # ------------------------------------------------------------------

    if delta_true !== nothing

        delta_true_fn = MMEGPD.get_delta(delta_true)

        delta_true_vals = [
            delta_true_fn(r)
            for r in r_grid
        ]

    end

    # ------------------------------------------------------------------
    # Choose the DISPLAY coordinate independently of the spline
    # coordinate.
    # ------------------------------------------------------------------

    if xscale == :natural

        x_grid = r_grid
        xlabel = "r"

    elseif xscale == :log

        x_grid = log.(r_grid)
        xlabel = "log(r)"

    else

        error(
            "xscale must be :natural or :log"
        )
    end

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    if plot_fn

        if delta_true !== nothing

            p = plot(
                x_grid,
                delta_true_vals,
                linewidth = 3,
                color = :black,
                label = "true δ(r)",
                xlabel = xlabel,
                ylabel = "δ(r)"
            )

        else

            p = plot(
                x_grid,
                delta_hat,
                linewidth = 3,
                color = :blue,
                label = "estimated δ(r)",
                xlabel = xlabel,
                ylabel = "δ(r)"
            )

        end

        # Confidence interval
        if show_ci

            plot!(
                p,
                x_grid,
                lower,
                fillrange = upper,
                fillalpha = 0.2,
                linealpha = 0,
                color = :blue,
                label = "$(round(Int, 100 * level))% CI"
            )

        end

        # Estimated spline
        plot!(
            p,
            x_grid,
            delta_hat,
            linewidth = 3,
            color = :blue,
            label = "estimated δ(r)"
        )

        return p
    end

    # ------------------------------------------------------------------
    # Return values for simulations / further analysis
    # ------------------------------------------------------------------

    return (
        r = r_grid,
        x = x_grid,
        estimate = delta_hat,
        lower = lower,
        upper = upper,
        se_logdelta = spline_ci.se_logdelta
    )
end