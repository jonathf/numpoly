"""Attempt to cast an object into a polynomial array."""
import numpy
import numpoly

from .compose import compose_polynomial_array


def polynomial(
        poly_like=None,
        names=None,
        dtype=None,
        allocation=None,
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
        allocation (Optional[int]):
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Returns:
        (numpoly.ndpoly):
            Polynomial based on input ``poly_like``.

    Examples:
        >>> numpoly.polynomial({(1,): 1})
        polynomial(q0)
        >>> q0, q1 = numpoly.variable(2)
        >>> q0**2+q0*q1+2
        polynomial(q0*q1+q0**2+2)
        >>> -3*q0+q0**2+q1
        polynomial(q0**2+q1-3*q0)
        >>> numpoly.polynomial([q0*q1, q0, q1])
        polynomial([q0*q1, q0, q1])
        >>> numpoly.polynomial([1, 2, 3])
        polynomial([1, 2, 3])
        >>> import sympy
        >>> q0_, q1_ = sympy.symbols("q0, q1")
        >>> numpoly.polynomial(3*q0_*q1_-4+q0_**5)
        polynomial(q0**5+3*q0*q1-4)

    """
    if poly_like is None:
        poly = numpoly.ndpoly(
            exponents=[(0,)],
            shape=(),
            names=names,
            dtype=dtype,
            allocation=allocation,
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
            allocation=allocation,
        )

    elif isinstance(poly_like, numpoly.ndpoly):
        if names is None:
            names = poly_like.names
        poly = numpoly.ndpoly.from_attributes(
            exponents=poly_like.exponents,
            coefficients=poly_like.coefficients,
            names=names,
            dtype=dtype,
            allocation=allocation,
        )

    # assume polynomial converted to structured array
    elif isinstance(poly_like, numpy.ndarray) and poly_like.dtype.names:

        keys = numpy.asarray(poly_like.dtype.names, dtype="U")
        keys = [key for key in keys if not key.isdigit()]
        keys = numpy.array(keys, dtype="U%d" % numpy.max(numpy.char.str_len(keys)))
        exponents = keys.view(numpy.uint32)-numpoly.ndpoly.KEY_OFFSET
        exponents = exponents.reshape(len(keys), -1)

        coefficients = [poly_like[key] for key in poly_like.dtype.names]
        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=names,
            allocation=allocation,
        )

    elif isinstance(poly_like, (int, float, numpy.ndarray, numpy.generic)):
        poly = numpoly.ndpoly.from_attributes(
            exponents=[(0,)],
            coefficients=numpy.array([poly_like]),
            names=names,
            dtype=dtype,
            allocation=allocation,
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
            allocation=allocation,
        )

    else:
        poly = compose_polynomial_array(
            arrays=poly_like,
            dtype=dtype,
            allocation=allocation,
        )

    return poly
