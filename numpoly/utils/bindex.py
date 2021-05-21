"""Multi indices for monomial exponents."""
from __future__ import annotations
from typing import Optional

import numpy
import numpy.typing

from .glexindex import glexindex


def bindex(
        start: numpy.typing.ArrayLike,
        stop: Optional[numpy.typing.ArrayLike] = None,
        dimensions: int = 1,
        ordering: str = "G",
        cross_truncation: numpy.typing.ArrayLike = 1.,
) -> numpy.ndarray:
    """
    Generate multi-indices for the monomial exponents.

    Args:
        start:
            The lower order of the indices. If array of int, counts as lower
            bound for each axis.
        stop:
            The maximum shape included. If omitted: stop <- start; start <- 0
            If int is provided, set as largest total order. If array of int,
            set as upper bound for each axis.
        dimensions:
            The number of dimensions in the expansion.
        ordering:
            Short hand for the criteria to sort the indices by.

            ``G``
                Graded sorting, meaning the indices are always sorted by the
                index sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will
                therefore be consider larger than both ``(3, 1, 1)`` and
                ``(1, 1, 3)``.
            ``R``
                Lexicographical sorting, meaning that ``(1, 3)`` is
                considered bigger than ``(3, 1)``, instead of the opposite.
            ``I``
                The biggest values are in the front instead of the back.
        cross_truncation:
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion. If two values are provided, first is low bound
            truncation, while the latter upper bound. If only one value, upper
            bound is assumed.

    Returns:
        Order indices.

    Examples:
        >>> numpoly.bindex(4).tolist()
        [[0], [1], [2], [3]]
        >>> numpoly.bindex(2, dimensions=2).tolist()
        [[0, 0], [0, 1], [1, 0]]
        >>> numpoly.bindex(start=2, stop=3, dimensions=2).tolist()
        [[0, 2], [1, 1], [2, 0]]
        >>> numpoly.bindex([1, 2, 3]).tolist()
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 2]]
        >>> numpoly.bindex([1, 2, 3], cross_truncation=numpy.inf).tolist()
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 2], [0, 1, 1], [0, 1, 2]]

    """
    ordering = ordering.upper()
    graded = "G" in ordering
    reverse = "R" not in ordering
    output = glexindex(
        start=start,
        stop=stop,
        dimensions=dimensions,
        cross_truncation=cross_truncation,
        graded=graded,
        reverse=reverse,
    )
    indices = slice(None, None, -1) if "I" in ordering else slice(None)
    return output[indices]
