"""Permute the dimensions of an array."""
import numpy
import numpoly

from .common import implements


@implements(numpy.transpose)
def transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    Args:
        a (numpoly.ndpoly):
            Input array.
        axes (int, Sequence[int]):
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

    Returns:
        (numpoly.ndpoly):
            `a` with its axes permuted.  A view is returned whenever possible.

    Examples:
        >>> poly = numpoly.monomial(3, names=("x", "y")).reshape(2, 3)
        >>> poly
        polynomial([[1, y, x],
                    [y**2, x*y, x**2]])
        >>> numpoly.transpose(poly)
        polynomial([[1, y**2],
                    [y, x*y],
                    [x, x**2]])

    """
    a = numpoly.aspolynomial(a)
    out = numpy.transpose(a.values, axes=axes)
    out = numpoly.polynomial(out, names=a.indeterminants)
    return out
