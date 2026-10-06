function difference_matrix(K, p)

    if p < 1
        throw(ArgumentError("Difference order p must be >= 1"))
    end

    if p >= K
        throw(ArgumentError("Difference order must be smaller than number of coefficients"))
    end

    nrows = K - p

    D = zeros(nrows, K)

    # finite difference coefficients
    coeffs = [
        (-1)^j * binomial(p, j)
        for j in 0:p
    ]

    for i in 1:nrows
        for j in 0:p
            D[i, i+j] = coeffs[j+1]
        end
    end

    return D
end