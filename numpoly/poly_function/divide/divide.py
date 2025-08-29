"""Return a true division of the inputs, element-wise."""

from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing

from ...baseclass import PolyLike, ndpoly
from ...dispatch import implements_function
from .divmod import poly_divmod


@implements_function(numpy.true_divide)
def poly_divide(
    x1: PolyLike,
    x2: PolyLike,
    out: Optional[ndpoly] = None,
    where: numpy.typing.ArrayLike = True,
    **kwargs: Any,
) -> ndpoly:
    """
    Return a polynomial division of the inputs, element-wise.

    Note that if divisor is a polynomial, then the division could have a
    remainder, as polynomial division is not exactly the same as numerical
    division.

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

    Return:
        This is a scalar if both `x1` and `x2` are scalars.

    Example:
        >>> q0 = numpoly.variable()
        >>> poly = numpoly.polynomial([14, q0**2-3])
        >>> numpoly.poly_divide(poly, 4)
        polynomial([3.5, 0.25*q0**2-0.75])
        >>> numpoly.poly_divide(poly, q0)
        polynomial([0.0, q0])

    """
    dividend, _ = poly_divmod(x1, x2, out=out, where=where, **kwargs)
    return dividend
