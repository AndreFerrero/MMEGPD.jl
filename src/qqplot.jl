function radial_qqplot(fit, probs_lim = [0.01, 0.9999], nprobs = 99)
  probs = collect(range(probs_lim[1], probs_lim[2], length=nprobs))

  q_empirical = quantile(fit.R, probs)

  if fit.bernstein

    q_model = [
        MMEGPD.egpd_quantile(p, fit.theta, fit.w)
        for p in probs
    ]
  else

    q_model = [
        MMEGPD.egpd_quantile(p, fit.theta)
        for p in probs
    ]
  end

  scatter(q_empirical, q_model;
      xlabel = "Empirical quantiles",
      ylabel = "Model quantiles",
      label = "Quantiles",
      legend = :topleft
  )

  q_min = min(
      minimum(q_empirical),
      minimum(q_model)
  )

  q_max = max(
      maximum(q_empirical),
      maximum(q_model)
  )

  plot!(
      [q_min, q_max],
      [q_min, q_max];
      linestyle = :dash,
      linewidth = 2,
      leg=false
  )
end