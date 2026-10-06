# This function main purpose is to provide a basis matrix given the data from the number of basis functions and the degree of the spline, in contrast to the package default behaviour which requires the order (degre + 1), so that it is more intutive to use if the user prefers thinking about degree of the spline.
# Breakpoints are equally spaced within the range of the data starting from zero

function bspline_basis(
    x;
    nbreaks=10,
    degree=3
)

    xmin = 0
    xmax = maximum(x)

    order = degree + 1

    breaks = range(
        xmin,
        xmax,
        length=nbreaks
    )

    basis = BSplineBasis(
        order,
        breaks
    )

    X = basismatrix(
        basis,
        x
    )

    return basis, X

end

function basis_matrix(
    basis::BSplines.BSplineBasis,
    x
)
    return BSplines.basismatrix(basis, x)
end