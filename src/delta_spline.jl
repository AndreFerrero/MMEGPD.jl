# This function main purpose is to construct the spline function for the delta function used inside the joint distribution, to evaluate the guassian term. The evaluation is done until the maximum observed value of the radius.
# The output is a function that takes the desired r value and returns the corresponding delta.

# Dispatcher with BSplines basis object
function delta_spline(
    basis,
    beta
)

    rmax = basis.breakpoints[end]

    return r -> begin

        r_eval =
            min(r,rmax)

        B =
            basis_matrix(
                basis,
                [r_eval]
            )

        eta =
            dot(
                B[1,:],
                beta
            )

        return exp(eta)

    end

end
