"""Attempt to cast an object into a polynomial array."""
import numpy
import numpoly

from .compose import compose_polynomial_array


def polynomial(
        poly_like=None,
        names=None,
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
    structured array    Assumes that the input are raw polynomial core and can
                        be used to construct a polynomial without changing the
                        data. Used for developer convenience.
    ==================  =======================================================

    Args:
        poly_like (typing.Any):
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        names (str, typing.Tuple[str, ...]):
            Name of the indeterminant variables. If possible to infer from
            ``poly_like``, this argument will be ignored.
        dtype (type, numpy.dtype):
            Data type used for the polynomial coefficients.

    Returns:
        (numpoly.ndpoly):
            Polynomial based on input ``poly_like``.

    Examples:
        >>> numpoly.polynomial({(1,): 1})
        polynomial(q)
        >>> x, y = numpoly.symbols("x y")
        >>> x**2 + x*y + 2
        polynomial(x**2+x*y+2)
        >>> -3*x + x**2 + y
        polynomial(-3*x+x**2+y)
        >>> numpoly.polynomial([x*y, x, y])
        polynomial([x*y, x, y])
        >>> numpoly.polynomial([1, 2, 3])
        polynomial([1, 2, 3])
        >>> import sympy
        >>> x_, y_ = sympy.symbols("x, y")
        >>> numpoly.polynomial(3*x_*y_ - 4 + x_**5)
        polynomial(x**5+3*x*y-4)

    """
    if poly_like is None:
        poly = numpoly.ndpoly(
            exponents=[(0,)],
            shape=(),
            names=names,
            dtype=dtype,
        )
        poly[";"] = 0

    elif isinstance(poly_like, dict):
        poly = numpoly.ndpoly(exponents=[(0,)], shape=())
        exponents, coefficients = zip(*list(poly_like.items()))
        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=names,
            dtype=dtype,
        )

    elif isinstance(poly_like, numpoly.ndpoly):
        if names is None:
            names = poly_like.names
        poly = numpoly.ndpoly.from_attributes(
            exponents=poly_like.exponents,
            coefficients=poly_like.coefficients,
            names=names,
            dtype=dtype,
        )

    # assume polynomial converted to structured array
    elif isinstance(poly_like, numpy.ndarray) and poly_like.dtype.names:

        keys = numpy.asarray(poly_like.dtype.names, dtype="U")
        exponents = keys.flatten().view(numpy.uint32)-numpoly.ndpoly.KEY_OFFSET
        exponents = exponents.reshape(len(keys), -1)

        coefficients = [poly_like[key] for key in poly_like.dtype.names]
        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=names,
        )

    elif isinstance(poly_like, (int, float, numpy.ndarray, numpy.generic)):
        poly = numpoly.ndpoly.from_attributes(
            exponents=[(0,)],
            coefficients=numpy.array([poly_like]),
            names=names,
            dtype=dtype,
        )

    # handler for sympy objects
    elif hasattr(poly_like, "as_poly"):
        poly_like = poly_like.as_poly()
        exponents = poly_like.monoms()
        coefficients = [int(coeff) if coeff.is_integer else float(coeff)
                        for coeff in poly_like.coeffs()]
        names = [str(elem) for elem in poly_like.gens]
        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=names,
        )

    else:
        poly = compose_polynomial_array(
            arrays=poly_like,
            dtype=dtype,
        )

    return poly
