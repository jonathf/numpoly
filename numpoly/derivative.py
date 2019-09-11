"""Polynomials differentiation functions."""
import numpy
import numpoly


def diff(poly, *diffvars):
    """
    Polynomial differential operator.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to differentiate.
        diffvars (numpoly.ndpoly, str):
            Singleton variables to take derivative off.

    Returns:
        Same as ``poly`` but differentiated with respect to ``diffvars``.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([1, x, x*y**2+1])
        >>> print(poly)
        [1 x 1+x*y**2]
        >>> print(numpoly.diff(poly, "x"))
        [0 1 y**2]
        >>> print(numpoly.diff(poly, x, y))
        [0 0 2*y]

    """
    poly = numpoly.polynomial(poly)

    for diffvar in diffvars:
        if isinstance(diffvar, str):
            idx = poly.names.index(diffvar)
        elif isinstance(diffvar, int):
            idx = diffvar
        else:
            diffvar = numpoly.polynomial(diffvar)
            assert len(diffvar.names) == 1, "only one at the time"
            assert numpy.all(diffvar.exponents == 1), (
                "derivative variable assumes singletons")
            idx = poly.names.index(diffvar.names[0])

        exponents = poly.exponents
        coefficients = [
            (exponent[idx]*coefficient.T).T
            for exponent, coefficient in zip(exponents, poly.coefficients)
        ]
        exponents[:, idx] -= 1
        if numpy.any(exponents < 0):
            indices = exponents[:, idx] == -1
            coefficients = [coefficient for coefficient, idx in zip(
                coefficients, indices) if not idx]
            exponents = numpy.delete(
                exponents, numpy.argwhere(indices), axis=0)

        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=poly.names,
        )
    return poly


def gradient(poly):
    """
    Polynomial gradient operator.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to differentiate. If polynomial vector is provided,
            the Jacobi-matrix is returned instead.

    Returns:
        Same as ``poly`` but with an extra first dimensions, one for each
        variable in ``poly.indeterminants``, filled with gradient values.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = 5*x**5+4
        >>> print(numpoly.gradient(poly))
        [25*x**4]
        >>> poly = 4*x**3+2*y**2+3
        >>> print(numpoly.gradient(poly))
        [12*x**2 4*y]
        >>> poly = numpoly.polynomial([1, x**3, x*y**2+1])
        >>> print(numpoly.gradient(poly))
        [[0 3*x**2 y**2]
         [0 0 2*x*y]]

    """
    polys = [diff(poly, diffvar)[numpy.newaxis]
             for diffvar in poly.indeterminants]
    return numpoly.concatenate(polys, axis=0)


def hessian(poly):
    """
    Construct Hessian matrix of polynomials.

    Make Hessian matrix out of a polynomial. Tensor is returned if polynomial
    is a vector.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to differentiate.

    Returns:
        Same as ``poly`` but with two extra dimensions, one for each
        variable in ``poly.indeterminants``, filled with Hessian values.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = 5*x**5+4
        >>> print(numpoly.hessian(poly))
        [[100*x**3]]
        >>> poly = 4*x**3+2*y**2+3
        >>> print(numpoly.hessian(poly))
        [[24*x 0]
         [0 4]]
        >>> poly = numpoly.polynomial([1, x, x*y**2+1])
        >>> print(numpoly.hessian(poly))
        [[[0 0 0]
          [0 0 2*y]]
        <BLANKLINE>
         [[0 0 2*y]
          [0 0 2*x]]]

    """
    return gradient(gradient(poly))
