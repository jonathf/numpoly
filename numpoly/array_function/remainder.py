"""Return element-wise remainder of division."""
import numpy
import numpoly


def remainder(x1, x2, out=None, where=True, **kwargs):
    """
    Return element-wise remainder of division.

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
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        (numpoly.ndpoly):
            The element-wise remainder of the quotient
            ``floor_divide(x1, x2)``. This is a scalar if both `x1` and `x2`
            are scalars.

    Notes:
        Unlike numbers, this returns the polynomial division and polynomial
        remainder. This means that this function is _not_ backwards compatible
        with ``numpy.remainder`` for constants. For example:
        ``numpy.remainder(11, 2) == 1`` while
        ``numpoly.remainder(11, 2) == 0``.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> denominator = [x*y**2+2*x**3*y**2, -2+x*y**2]
        >>> numerator = -2+x*y**2
        >>> numpoly.remainder(denominator, numerator)
        polynomial([2.0+4.0*x**2, 0.0])

    """
    dividend, remainder = numpoly.divmod(x1, x2, out=out, where=where, **kwargs)
    return remainder
