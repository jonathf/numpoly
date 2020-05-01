"""Return element-wise remainder of division."""
import numpy
import numpoly

from .common import implements


@implements(numpy.mod)
def mod(x1, x2, out=None, where=True, **kwargs):
    """
    Return element-wise remainder of division.

    Computes the remainder complementary to the `floor_divide` function.  It is
    equivalent to the Python modulus operator``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to ``np.remainder`` is
    ``mod``.

    Note that this should not be confused with:

    * Python 3.7's `math.remainder` and C's ``remainder``, which
    computes the IEEE remainder, which are the complement to
    ``round(x1 / x2)``.
    * The MATLAB ``rem`` function and or the C ``%`` operator which is the
    complement to ``int(x1 / x2)``.

    Args:
        x1 (numpoly.ndpoly):
            Dividend array.
        x2 (numpoly.ndpoly):
            Divisor array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the
            output).
        out (Optional[numpy.ndarray]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (Union[bool, numpy.ndarray]):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        **kwargs
            Keyword args passed to numpy.ufunc.

    Returns:
        (numpoly.ndpoly):
            The element-wise remainder of the quotient
            ``floor_divide(x1, x2)``. This is a scalar if both `x1` and `x2`
            are scalars.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> denominator = [x*y**2+2*x**3*y**2, -2+x*y**2]
        >>> numerator = -2+x*y**2
        >>> numpoly.mod(denominator, numerator)
        polynomial([2.0+4.0*x**2, 0.0])

    """
    dividend, remainder = numpoly.poly_divide(x1, x2, out=out, where=where, **kwargs)
    return remainder
