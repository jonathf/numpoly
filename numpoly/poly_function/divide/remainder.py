"""Return element-wise remainder of polynomial division."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing

from ...baseclass import ndpoly, PolyLike
from ...dispatch import implements_function
from .divmod import poly_divmod


@implements_function(numpy.remainder)
def poly_remainder(
        x1: PolyLike,
        x2: PolyLike,
        out: Optional[ndpoly] = None,
        where: numpy.typing.ArrayLike = True,
        **kwargs: Any,
) -> ndpoly:
    """
    Return element-wise remainder of polynomial division.

    Args:
        x1:
            Dividend array.
        x2:
            Divisor array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the
            output).
        out:
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where:
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        The element-wise remainder of the quotient ``floor_divide(x1, x2)``.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes:
        Unlike numbers, this returns the polynomial division and polynomial
        remainder. This means that this function is _not_ backwards compatible
        with ``numpy.remainder`` for constants. For example:
        ``numpoly.remainder(11, 2) == 1`` while
        ``numpoly.poly_remainder(11, 2) == 0``.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> denominator = [q0*q1**2+2*q0**3*q1**2, -2+q0*q1**2]
        >>> numerator = -2+q0*q1**2
        >>> numpoly.poly_remainder(denominator, numerator)
        polynomial([4.0*q0**2+2.0, 0.0])

    """
    dividend, remainder = poly_divmod(x1, x2, out=out, where=where, **kwargs)
    return remainder
