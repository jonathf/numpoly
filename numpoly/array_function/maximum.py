"""Element-wise maximum of array elements."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.maximum)
def maximum(
    x1: PolyLike,
    x2: PolyLike,
    out: Optional[ndpoly] = None,
    **kwargs: Any,
) -> ndpoly:
    """
    Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Args:
        x1, x2:
            The arrays holding the elements to be compared. If ``x1.shape !=
            x2.shape``, they must be broadcastable to a common shape (which
            becomes the shape of the output).
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
        The maximum of `x1` and `x2`, element-wise. This is a scalar if
        both `x1` and `x2` are scalars.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.maximum(3, 4)
        polynomial(4)
        >>> numpoly.maximum(4*q0, 3*q0)
        polynomial(4*q0)
        >>> numpoly.maximum(q0, q1)
        polynomial(q1)
        >>> numpoly.maximum(q0**2, q0)
        polynomial(q0**2)
        >>> numpoly.maximum([1, q0, q0**2, q0**3], q1)
        polynomial([q1, q1, q0**2, q0**3])
        >>> numpoly.maximum(q0+1, q0-1)
        polynomial(q0+1)

    """
    del out
    x1, x2 = numpoly.align_polynomials(x1, x2)
    coefficients1 = x1.coefficients
    coefficients2 = x2.coefficients
    out_ = numpy.zeros(x1.shape, dtype=bool)

    options = numpoly.get_options()
    for idx in numpoly.glexsort(x1.exponents.T, graded=options["sort_graded"],
                                reverse=options["sort_reverse"]):

        indices = (coefficients1[idx] != 0) | (coefficients2[idx] != 0)
        indices &= coefficients1[idx] != coefficients2[idx]
        out_[indices] = (coefficients1[idx] > coefficients2[idx])[indices]
    return numpoly.where(out_, x1, x2)
