"""Return the truth value of (x1 > x2) element-wise."""

from __future__ import annotations
from typing import Any, Optional

import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.greater)
def greater(
    x1: PolyLike,
    x2: PolyLike,
    out: Optional[numpy.ndarray] = None,
    **kwargs: Any,
) -> numpy.ndarray:
    """
    Return the truth value of (x1 > x2) element-wise.

    Args:
        x1, x2:
            Input arrays. If ``x1.shape != x2.shape``, they must be
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
        Output array, element-wise comparison of `x1` and `x2`. Typically of
        type bool, unless ``dtype=object`` is passed. This is a scalar if both
        `x1` and `x2` are scalars.

    Example:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.greater(3, 4)
        False
        >>> numpoly.greater(4*q0, 3*q0)
        True
        >>> numpoly.greater(q0, q1)
        False
        >>> numpoly.greater(q0**2, q0)
        True
        >>> numpoly.greater([1, q0, q0**2, q0**3], q1)
        array([False, False,  True,  True])
        >>> numpoly.greater(q0+1, q0-1)
        True
        >>> numpoly.greater(q0, q0)
        False

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    coefficients1 = x1.coefficients
    coefficients2 = x2.coefficients
    if out is None:
        out = numpy.greater(coefficients1[0], coefficients2[0], **kwargs)
    if not out.shape:
        return greater(x1.ravel(), x2.ravel(), out=out.ravel()).item()

    options = numpoly.get_options()
    for idx in numpoly.glexsort(
        x1.exponents.T, graded=options["sort_graded"], reverse=options["sort_reverse"]
    ):
        indices = (coefficients1[idx] != 0) | (coefficients2[idx] != 0)
        indices &= coefficients1[idx] != coefficients2[idx]
        out[indices] = numpy.greater(coefficients1[idx], coefficients2[idx], **kwargs)[
            indices
        ]
    return out
