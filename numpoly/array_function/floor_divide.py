"""Return the largest integer smaller or equal to the division of the inputs."""
import numpy
import numpoly

from .common import implements


@implements(numpy.floor_divide)
def floor_divide(x1, x2, out=None, where=True, **kwargs):
    """
    Return the largest integer smaller or equal to the division of the inputs.

    It is equivalent to the Python ``//`` operator and pairs with the
    Python ``%`` (`remainder`), function so that ``a = a % b + b * (a // b)``
    up to roundoff.

    Args:
        x1 (numpoly.ndpoly):
            Numerator.
        x2 (numpoly.ndpoly):
            Denominator. If ``x1.shape != x2.shape``, they must be
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

    Examples:
        >>> xyz = [1, 2, 4]*numpoly.symbols("x y z")
        >>> numpoly.floor_divide(xyz, 2.)
        polynomial([0.0, y, 2.0*z])
        >>> numpoly.floor_divide(xyz, [1, 2, 4])
        polynomial([x, y, z])
        >>> numpoly.floor_divide([1, 2, 4], xyz)
        Traceback (most recent call last):
            ...
        ValueError: only constant polynomials can be converted to array.

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    x2 = x2.tonumpy()
    no_output = out is None
    if no_output:
        out = numpoly.ndpoly(
            exponents=x1.exponents,
            shape=x1.shape,
            names=x1.indeterminants,
            dtype=numpy.common_type(x1, numpy.array(1.)),
        )
    for key in x1.keys:
        numpy.floor_divide(x1[key], x2, out=out[key], where=where, **kwargs)
    if no_output:
        out = numpoly.clean_attributes(out)
    return out


# def polynomial_division(x1, x2):
#     """
#     Examples:
#         >>> # polynomial_division(numpoly.monomial(5, names="x"), numpoly.symbols("x"))
#     """
#     x1, x2 = numpoly.align_polynomials(x1, x2)
#     out = 0
#     for idx, name in enumerate(x1.names):
#         pass


#     exponents = x1.exponents.copy()
#     coefficients = x1.coefficients.copy()
#     orders1 = numpy.sum(x1.exponents, -1)
#     orders2 = numpy.sum(x2.exponents, -1)

#     while numpy.max(orders1) > numpy.max(orders2):
#         exponents
