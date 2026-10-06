function as_pairs(x)

    if x isa AbstractVector

        return hcat(
            x[1:end-1],
            x[2:end]
        )

    elseif x isa AbstractMatrix

        @assert size(x, 2) == 2 "Matrix input must have exactly two columns."

        return x

    else

        throw(ArgumentError(
            "x must be a vector or an N×2 matrix"
        ))

    end

end