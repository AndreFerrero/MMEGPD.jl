struct CRSplineBasis{T}
    breakpoints::Vector{T}
    D::Matrix{T}
    B::Matrix{T}
    F::Matrix{T}
    S::Matrix{T}
end

function crspline_basis(
    x;
    nbreaks::Int=10,
    knot_method::Symbol=:even,
)

    if nbreaks < 3
        throw(ArgumentError(
            "A natural cubic regression spline requires at least 3 breakpoints."
        ))
    end

    xmin = minimum(x)
    xmax = maximum(x)

    # ------------------------------------------------------------
    # Choose breakpoints
    # ------------------------------------------------------------

    if knot_method === :even

        breakpoints = collect(
            range(xmin, xmax, length=nbreaks)
        )

    elseif knot_method === :quantile

        probs = range(0.0, 1.0, length=nbreaks)

        breakpoints = collect(quantile(x, probs))

        if length(unique(breakpoints)) < nbreaks
            throw(ArgumentError(
                "Quantile knot placement produced duplicate breakpoints. " *
                "Try fewer breakpoints or use knot_method=:even."
            ))
        end

    else
        throw(ArgumentError(
            "Unknown knot_method=$knot_method. " *
            "Use :even or :quantile."
        ))
    end

    K = length(breakpoints)

    h = diff(breakpoints)

    # ------------------------------------------------------------
    # D matrix: (K-2) × K
    # ------------------------------------------------------------

    D = zeros(eltype(breakpoints), K - 2, K)

    for i in 1:(K-2)

        D[i, i] = 1.0 / h[i]

        D[i, i+1] = -1.0 / h[i] - 1.0 / h[i+1]

        D[i, i+2] = 1.0 / h[i+1]

    end

    # ------------------------------------------------------------
    # B matrix: (K-2) × (K-2)
    # ------------------------------------------------------------

    B = zeros(eltype(breakpoints), K - 2, K - 2)

    for i in 1:(K-2)

        B[i, i] = (h[i] + h[i+1]) / 3.0

        if i < K - 2
            B[i, i+1] = h[i+1] / 6.0
            B[i+1, i] = h[i+1] / 6.0
        end

    end

    # ------------------------------------------------------------
    # F such that delta = F * beta
    # ------------------------------------------------------------

    F = zeros(eltype(breakpoints), K, K)

    F[2:(K-1), :] = B \ D

    # ------------------------------------------------------------
    # Integrated squared curvature penalty
    # ------------------------------------------------------------

    S = D' * (B \ D)

    return CRSplineBasis(breakpoints, D, B, F, S)
end

function cr_basis_row(
    basis::CRSplineBasis,
    x
)

    breakpoints = basis.breakpoints
    F = basis.F

    K = length(breakpoints)

    # Keep x inside the knot range
    x_eval = clamp(
        Float64(x),
        breakpoints[1],
        breakpoints[end]
    )

    # Find interval [x_j, x_{j+1}]
    j = searchsortedlast(breakpoints, x_eval)

    # At the right boundary, use the final interval
    j = min(j, K - 1)

    xj = breakpoints[j]
    xjp1 = breakpoints[j+1]

    h = xjp1 - xj

    # ------------------------------------------------------------
    # a^- and a^+
    # ------------------------------------------------------------

    a_minus =
        (xjp1 - x_eval) / h

    a_plus =
        (x_eval - xj) / h

    # ------------------------------------------------------------
    # c^- and c^+
    # ------------------------------------------------------------

    c_minus =
        ((xjp1 - x_eval)^3 / h -
         h * (xjp1 - x_eval)) / 6.0

    c_plus =
        ((x_eval - xj)^3 / h -
         h * (x_eval - xj)) / 6.0

    # ------------------------------------------------------------
    # Construct b(x)
    #
    # f(x) =
    # a^- beta_j
    # + a^+ beta_{j+1}
    # + c^- delta_j
    # + c^+ delta_{j+1}
    #
    # delta = F beta
    # ------------------------------------------------------------

    b = zeros(K)

    b[j] += a_minus
    b[j+1] += a_plus

    b .+= c_minus .* F[j, :]
    b .+= c_plus .* F[j+1, :]

    return b
end

function cr_basismatrix(
    basis::CRSplineBasis,
    x
)

    X = Matrix{Float64}(
        undef,
        length(x),
        length(basis.breakpoints)
    )

    for i in eachindex(x)
        X[i, :] .= cr_basis_row(
                basis,
                x[i]
            )
    end

    return X
end

function basis_matrix(
    basis::CRSplineBasis,
    x
)
    return cr_basismatrix(basis, x)
end