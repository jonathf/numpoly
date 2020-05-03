"""Return a true division of the inputs, element-wise."""
from __future__ import division

import numpy
import numpoly

from .common import implements


@implements(numpy.divide, numpy.true_divide)
def true_divide(x1, x2, out=None, where=True, **kwargs):
    """
    Return a true division of the inputs, element-wise.

    Instead of the Python traditional 'floor division', this returns a true
    division. True division adjusts the output type to present the best
    answer, regardless of input types.

    Note that if divisor is a polynomial, then the division could have a
    remainder, as polynomial division is not exactly the same as numerical
    division.

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
        where (Optional[numpy.ndarray]):
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
            This is a scalar if both `x1` and `x2` are scalars.

    Examples:
        >>> x = numpoly.symbols("x")
        >>> poly = numpoly.polynomial([14, x**2-3])
        >>> numpoly.true_divide(poly, 4)
        polynomial([3.5, -0.75+0.25*x**2])
        >>> numpoly.true_divide(poly, x)
        polynomial([0.0, x])

    """
    dividend, remainder = numpoly.divmod(x1, x2, out=out, where=where, **kwargs)
    return dividend
