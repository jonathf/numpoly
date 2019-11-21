"""Attempt to cast an object into a polynomial array."""
import numpy
import numpoly

from .compose import compose_polynomial_array


def polynomial(
        poly_like=None,
        indeterminants=None,
        dtype=None,
):
    """
    Attempt to cast an object into a polynomial array.

    Supports various casting options:

    ==================  =======================================================
    ``dict``            Keys are tuples that represent polynomial exponents,
                        and values are numpy arrays that represents polynomial
                        coefficients.
    ``numpoly.ndpoly``  Copy of the polynomial.
    ``numpy.ndarray``   Constant term polynomial.
    ``sympy.Poly``      Convert polynomial from ``sympy`` to ``numpoly``,
                        if possible.
    ``Iterable``        Multivariate array construction.
    structured array    Assumes that the raw polynomial core. Used for
                        developer convenience.
    ==================  =======================================================

    Args:
        poly_like (typing.Any):
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        indeterminants (str, typing.Tuple[str, ...]):
            Name of the indeterminant variables. If possible to infer from
            ``poly_like``, this argument will be ignored.
        dtype (type, numpy.dtype):
            Data type used for the polynomial coefficients.

    Returns:
        (numpoly.ndpoly):
            Polynomial based on input ``poly_like``.

    Examples:
        >>> print(numpoly.polynomial({(1,): 1}))
        q
        >>> x, y = numpoly.symbols("x y")
        >>> print(x**2 + x*y + 2)
        x**2+x*y+2
        >>> print(-3*x + x**2 + y)
        -3*x+x**2+y
        >>> print(numpoly.polynomial([x*y, x, y]))
        [x*y x y]
        >>> print(numpoly.polynomial([1, 2, 3]))
        [1 2 3]
        >>> import sympy
        >>> x_, y_ = sympy.symbols("x, y")
        >>> print(numpoly.polynomial(3*x_*y_ - 4 + x_**5))
        x**5+3*x*y-4

    """
    if poly_like is None:
        poly = numpoly.ndpoly(
            exponents=[(0,)],
            shape=(),
            indeterminants=indeterminants,
            dtype=dtype,
        )
        poly["0"] = 0

    elif isinstance(poly_like, dict):
        poly = numpoly.ndpoly(exponents=[(0,)], shape=())
        exponents, coefficients = zip(*list(poly_like.items()))
        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
            dtype=dtype,
        )

    elif isinstance(poly_like, numpoly.ndpoly):
        if indeterminants is None:
            indeterminants = poly_like.names
        poly = numpoly.ndpoly.from_attributes(
            exponents=poly_like.exponents,
            coefficients=poly_like.coefficients,
            indeterminants=indeterminants,
            dtype=dtype,
        )

    # assume polynomial converted to structured array
    elif isinstance(poly_like, numpy.ndarray) and poly_like.dtype.names:

        keys = numpy.asarray(poly_like.dtype.names, dtype="U")
        exponents = keys.flatten().view(numpy.uint32)-48
        exponents = exponents.reshape(len(keys), -1)

        coefficients = [poly_like[key] for key in poly_like.dtype.names]
        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
        )

    elif isinstance(poly_like, (int, float, numpy.ndarray, numpy.generic)):
        poly = numpoly.ndpoly.from_attributes(
            exponents=[(0,)],
            coefficients=numpy.array([poly_like]),
            indeterminants=indeterminants,
            dtype=dtype,
        )

    # handler for sympy objects
    elif hasattr(poly_like, "as_poly"):
        poly_like = poly_like.as_poly()
        exponents = poly_like.monoms()
        coefficients = [int(coeff) if coeff.is_integer else float(coeff)
                        for coeff in poly_like.coeffs()]
        indeterminants = [str(elem) for elem in poly_like.gens]
        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
        )

    else:
        poly = compose_polynomial_array(
            arrays=poly_like,
            dtype=dtype,
        )

    return poly
