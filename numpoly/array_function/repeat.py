"""Repeat elements of an array."""
from __future__ import annotations

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.repeat)
def repeat(
    a: PolyLike,
    repeats: numpy.typing.ArrayLike,
    axis: int = 0,
) -> ndpoly:
    """
    Repeat elements of an array.

    Args:
        a:
            Input array.
        repeats:
            The number of repetitions for each element. `repeats` is
            broadcasted to fit the shape of the given axis.
        axis:
            The axis along which to repeat values. By default, use the
            flattened input array, and return a flat output array.

    Returns:
        Output array which has the same shape as `a`, except along the
        given axis.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.repeat(q0, 4)
        polynomial([q0, q0, q0, q0])
        >>> poly = numpoly.polynomial([[1, q0-1], [q0**2, q0]])
        >>> numpoly.repeat(poly, 2)
        polynomial([[1, q0-1],
                    [1, q0-1],
                    [q0**2, q0],
                    [q0**2, q0]])
        >>> numpoly.repeat(poly, 3, axis=1)
        polynomial([[1, 1, 1, q0-1, q0-1, q0-1],
                    [q0**2, q0**2, q0**2, q0, q0, q0]])
        >>> numpoly.repeat(poly, [1, 2], axis=0)
        polynomial([[1, q0-1],
                    [q0**2, q0],
                    [q0**2, q0]])

    """
    a = numpoly.aspolynomial(a)
    repeats = numpy.asarray(repeats)
    result = numpy.repeat(a.values, repeats=repeats, axis=axis)
    return numpoly.aspolynomial(result, names=a.indeterminants)
