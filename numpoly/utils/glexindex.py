"""Multi indices for monomial exponents."""
import numpy

from .cross_truncation import cross_truncate
from .glexsort import glexsort


def glexindex(start, stop=None, dimensions=1, cross_truncation=1.,
              graded=False, reverse=False):
    """
    Generate graded lexicographical multi-indices for the monomial exponents.

    Args:
        start (Union[int, numpy.ndarray]):
            The lower order of the indices. If array of int, counts as lower
            bound for each axis.
        stop (Union[int, numpy.ndarray, None]):
            The maximum shape included. If omitted: stop <- start; start <- 0
            If int is provided, set as largest total order. If array of int,
            set as upper bound for each axis.
        dimensions (int):
            The number of dimensions in the expansion.
        cross_truncation (float, Tuple[float, float]):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion. If two values are provided, first is low bound
            truncation, while the latter upper bound. If only one value, upper
            bound is assumed.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will therefore be
            consider larger than both ``(3, 1, 1)`` and ``(1, 1, 3)``.
        reverse (bool):
            Reversed lexicographical sorting meaning that ``(1, 3)`` is
            considered smaller than ``(3, 1)``, instead of the opposite.

    Returns:
        list:
            Order list of indices.

    Examples:
        >>> numpoly.glexindex(4).tolist()
        [[0], [1], [2], [3]]
        >>> numpoly.glexindex(2, dimensions=2).tolist()
        [[0, 0], [1, 0], [0, 1]]
        >>> numpoly.glexindex(start=2, stop=3, dimensions=2).tolist()
        [[2, 0], [1, 1], [0, 2]]
        >>> numpoly.glexindex([1, 2, 3]).tolist()
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]]
        >>> numpoly.glexindex([1, 2, 3], cross_truncation=numpy.inf).tolist()
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2], [0, 1, 2]]

    """
    if stop is None:
        start, stop = 0, start
    start = numpy.array(start, dtype=int).flatten()
    stop = numpy.array(stop, dtype=int).flatten()
    start, stop, _ = numpy.broadcast_arrays(start, stop, numpy.empty(dimensions))

    cross_truncation = cross_truncation*numpy.ones(2)
    indices = _glexindex(start, stop, cross_truncation)

    if indices.size:
        indices = indices[glexsort(indices.T, graded=graded, reverse=reverse)]

    return indices


def _glexindex(start, stop, cross_truncation=1.):
    """Backend for the glexindex function."""
    # At the beginning the current list of indices just ranges over the
    # last dimension.
    bound = stop.max()
    dimensions = len(start)
    start = numpy.clip(start, a_min=0, a_max=None)
    dtype = numpy.uint8 if bound < 256 else numpy.uint16
    range_ = numpy.arange(bound, dtype=dtype)
    indices = range_[:, numpy.newaxis]

    for idx in range(dimensions-1):

        # Truncate at each step to keep memory usage low
        if idx:
            indices = indices[cross_truncate(indices, bound-1, cross_truncation[1])]

        # Repeats the current set of indices.
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = numpy.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices)//bound)[:, numpy.newaxis]

        # Puts them two together.
        indices = numpy.column_stack((front, indices))

    # Complete the truncation scheme
    if dimensions == 1:
        indices = indices[(indices >= start) & (indices < bound)]
    else:
        lower = cross_truncate(indices, start-1, cross_truncation[0])
        upper = cross_truncate(indices, stop-1, cross_truncation[1])
        indices = indices[lower^upper]

    return numpy.array(indices, dtype=int).reshape(-1, dimensions)
