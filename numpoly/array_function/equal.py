"""Return (x1 == x2) element-wise."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.equal)
def equal(
        x1: PolyLike,
        x2: PolyLike,
        out: Optional[numpy.ndarray] = None,
        where: numpy.typing.ArrayLike = True,
        **kwargs: Any,
) -> numpy.ndarray:
    """
    Return (x1 == x2) element-wise.

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
            Elsewhere, the `out` array will retain its original value.
            Note that if an uninitialized `out` array is created via the
            default ``out=None``, locations within it where the condition is
            False will remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        Output array, element-wise comparison of `x1` and `x2`. Typically of
        type bool, unless ``dtype=object`` is passed. This is a scalar if both
        `x1` and `x2` are scalars.

    Examples:
        >>> q0, q1, q2 = q0q1q2 = numpoly.variable(3)
        >>> numpoly.equal(q0q1q2, q0)
        array([ True, False, False])
        >>> numpoly.equal(q0q1q2, [q1, q1, q2])
        array([False,  True,  True])
        >>> numpoly.equal(q0, q1)
        False

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    if out is None:
        out = numpy.ones(x1.shape, dtype=bool)
    if not out.shape:
        return equal(x1.ravel(), x2.ravel(), out=out.ravel()).item()
    for coeff1, coeff2 in zip(x1.coefficients, x2.coefficients):
        out &= numpy.equal(coeff1, coeff2,
                           where=numpy.asarray(where), **kwargs)
    return out
