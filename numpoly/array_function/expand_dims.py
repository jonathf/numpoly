"""Expand the shape of an array."""
import numpy
import numpoly

from ..dispatch import implements

@implements(numpy.expand_dims)
def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Args:
        a (numpoly.ndarray):
            Input array.
        axis (int):
            Position in the expanded axes where the new axis is placed.

    Returns:
        (ndpoly):
            View of `a` with the number of dimensions increased by one.

    Examples:
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
