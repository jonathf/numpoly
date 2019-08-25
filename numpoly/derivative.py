from . import align, baseclass, construct

import numpy


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
    poly = construct.polynomial(poly)

    for diffvar in diffvars:
        if isinstance(diffvar, str):
            idx = poly._indeterminants.index(diffvar)
        elif isinstance(diffvar, int):
            idx = diffvar
        else:
            diffvar = construct.polynomial(diffvar)
            assert len(diffvar._indeterminants) == 1, "only one at the time"
            assert numpy.all(diffvar.exponents == 1), (
                "derivative variable assumes singletons")
            idx = poly._indeterminants.index(diffvar._indeterminants[0])

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

        poly = construct.polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=poly._indeterminants,
        )

    return poly

