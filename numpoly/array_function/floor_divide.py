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
            When denominator is not a constant, floor-division is not possible.

    Examples:
        >>> numpoly.floor_divide([1, 3, 5], 2)
        polynomial([0, 1, 2])
        >>> xyz = [1, 2, 4]*numpoly.symbols("x y z")
        >>> numpoly.floor_divide(xyz, 2.)
        polynomial([0, y, 2*z])
        >>> numpoly.floor_divide(xyz, [1, 2, 4])
        polynomial([x, y, z])
        >>> numpoly.floor_divide([1, 2, 4], xyz)
        Traceback (most recent call last):
            ...
        ValueError: only constant polynomials can be converted to array.

    """
    x2 = numpoly.aspolynomial(x2)
    dividend, remainder = numpoly.divmod(
        x1, x2.tonumpy(), out=out, where=where, **kwargs)
    return dividend.astype(int)
