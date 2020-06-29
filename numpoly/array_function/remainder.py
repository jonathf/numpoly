"""Return element-wise remainder of division."""
import numpy
import numpoly

from ..dispatch import implements_ufunc

REMAINDER_ERROR_MSG = """\
Polynomial division differs from numerical division;
Use ``numpoly.poly_remainder`` to get polynomial remainder."""


@implements_ufunc(numpy.remainder)
def remainder(x1, x2, out=None, where=True, **kwargs):
    """
    Return element-wise remainder of numerical division.

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

    Examples:
        >>> numpoly.remainder([14, 7], 5)
        polynomial([4, 2])
        >>> numpoly.remainder(numpoly.variable(), 2) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        numpoly.baseclass.FeatureNotSupported: Polynomial division ...
        Use ``numpoly.poly_remainder`` to get polynomial remainder.

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    if not x1.isconstant() or not x2.isconstant():
        raise numpoly.FeatureNotSupported(REMAINDER_ERROR_MSG)
    return numpoly.polynomial(numpy.remainder(
        x1.tonumpy(), x2.tonumpy(), out=out, where=where, **kwargs))
