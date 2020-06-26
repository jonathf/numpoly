"""Return the largest integer smaller or equal to the division of the inputs."""
import numpy
import numpoly

from ..dispatch import implements

DIVIDE_ERROR_MSG = """
Divisor in division is a polynomial.
Polynomial division differs from numerical division;
Use ``numpoly.poly_divide`` to get polynomial division."""


@implements(numpy.floor_divide)
def floor_divide(x1, x2, out=None, where=True, **kwargs):
    """
    Return the largest integer smaller or equal to the division of the inputs.

    It is equivalent to the Python ``//`` operator and pairs with the
    Python ``%`` (`remainder`), function so that ``a = a % b + b * (a // b)``
    up to roundoff.

    Args:
        x1 (numpoly.ndpoly):
            Dividend.
        x2 (numpoly.ndpoly):
            Divisor. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the
            output).
        out (Optional[numpy.ndarray]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (Optional[numpy.ndarray]):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value.
            Note that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        (numpoly.ndpoly):
            This is a scalar if both `x1` and `x2` are scalars.

    Raises:
        ValueError:
            If `x2` contains indeterminants, numerical division is no longer
            possible and an error is raised instead. For polynomial
            division see ``numpoly.poly_divide``.

    Examples:
        >>> numpoly.floor_divide([1, 3, 5], 2)
        polynomial([0, 1, 2])
        >>> poly = [1, 2, 4]*numpoly.variable(3)
        >>> poly
        polynomial([q0, 2*q1, 4*q2])
        >>> numpoly.floor_divide(poly, 2.)
        polynomial([0.0, q1, 2.0*q2])
        >>> numpoly.floor_divide(poly, [1, 2, 4])
        polynomial([q0, q1, q2])

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    if not x2.isconstant():
        raise numpoly.FeatureNotSupported(DIVIDE_ERROR_MSG)
    x2 = x2.tonumpy()
    dtype=numpy.common_type(x1, x2)
    if x1.dtype == x2.dtype == "int64":
        dtype = "int64"
    no_output = out is None
    if no_output:
        out = numpoly.ndpoly(
            exponents=x1.exponents,
            shape=x1.shape,
            names=x1.indeterminants,
            dtype=dtype,
        )
    elif not isinstance(out, numpy.ndarray):
        assert len(out) == 1, "only one output"
        out = out[0]
    for key in x1.keys:
        out[key] = 0
        numpy.floor_divide(x1[key], x2, out=out[key], where=where, **kwargs)
    if no_output:
        out = numpoly.clean_attributes(out)
    return out
