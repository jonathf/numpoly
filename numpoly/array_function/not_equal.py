"""Return (x1 != x2) element-wise."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.not_equal)
def not_equal(
        x1: PolyLike,
        x2: PolyLike,
        out: Optional[numpy.ndarray] = None,
        where: numpy.typing.ArrayLike = True,
        **kwargs: Any,
) -> numpy.ndarray:
    """
    Return (x1 != x2) element-wise.

    Args:
        x1, x2:
            Input arrays.  If ``x1.shape != x2.shape``, they must be
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
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.not_equal([q0, q0], [q0, q1])
        array([False,  True])
        >>> numpoly.not_equal([q0, q0], [[q0, q1], [q1, q0]])
        array([[False,  True],
               [ True, False]])

    """
    x1, x2 = numpoly.align_exponents(x1, x2)
    if not x1.flags["OWNDATA"]:
        x1 = numpoly.polynomial(x1)
    if not x2.flags["OWNDATA"]:
        x2 = numpoly.polynomial(x2)
    # x1, x2 = numpoly.align_polynomials(x1, x2)
    where = numpy.asarray(where)
    for key in x1.keys:
        tmp = numpy.not_equal(
            x1.values[key], x2.values[key], where=where, **kwargs)
        if out is None:
            out = tmp
        else:
            out |= tmp
    return numpy.asarray(out)
