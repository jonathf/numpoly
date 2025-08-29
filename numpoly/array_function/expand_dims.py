"""Expand the shape of an array."""

from __future__ import annotations
import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.expand_dims)
def expand_dims(a: PolyLike, axis: int) -> ndpoly:
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Args:
        a:
            Input array.
        axis:
            Position in the expanded axes where the new axis is placed.

    Return:
        View of `a` with the number of dimensions increased by one.

    Example:
        >>> poly = numpoly.variable(2)
        >>> numpoly.expand_dims(poly, axis=0)
        polynomial([[q0, q1]])
        >>> numpoly.expand_dims(poly, axis=1)
        polynomial([[q0],
                    [q1]])

    """
    a = numpoly.aspolynomial(a)
    out = numpy.expand_dims(a.values, axis=axis)
    return numpoly.polynomial(out, names=a.indeterminants)
