"""Construct an array by repeating A the number of times given by reps."""
from __future__ import annotations

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.tile)
def tile(A: PolyLike, reps: numpy.typing.ArrayLike) -> ndpoly:
    """
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Args:
        A:
            The input array.
        reps:
            The number of repetitions of `A` along each axis.

    Returns:
        The tiled output array.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.tile(q0, 4)
        polynomial([q0, q0, q0, q0])
        >>> poly = numpoly.polynomial([[1, q0-1], [q0**2, q0]])
        >>> numpoly.tile(poly, 2)
        polynomial([[1, q0-1, 1, q0-1],
                    [q0**2, q0, q0**2, q0]])
        >>> numpoly.tile(poly, [2, 1])
        polynomial([[1, q0-1],
                    [q0**2, q0],
                    [1, q0-1],
                    [q0**2, q0]])

    """
    A = numpoly.aspolynomial(A)
    result = numpy.tile(A.values, reps=reps)
    return numpoly.aspolynomial(result, names=A.indeterminants)
