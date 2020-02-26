"""Multi indices for monomial exponents."""
import numpy

from .cross_truncation import cross_truncate


def bindex(start, stop=None, dimensions=1, ordering="G", cross_truncation=1.):
    """
    Generate multi-indices for the monomial exponents.

    Args:
        start (Union[int, numpy.ndarray]):
            The lower order of the indices. If array of int, counts as lower
            bound for each axis.
        stop (Union[int, numpy.ndarray, None]):
            The maximum shape included. If omitted: stop <- start; start <- 0
            If int is provided, set as largest total order. If array of int,
            set as upper bound for each axis.
        dimensions (int):
            The number of dimensions in the expansion
        ordering (str):
            Criteria to sort the indices by.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Returns:
        list:
            Order list of indices.

    Examples:
        >>> bindex(4).tolist()
        [[0], [1], [2], [3]]
        >>> bindex(2, dimensions=2).tolist()
        [[0, 0], [0, 1], [1, 0]]
        >>> bindex(start=2, stop=3, dimensions=2).tolist()
        [[0, 2], [1, 1], [2, 0]]
        >>> bindex(start=0, stop=2, dimensions=3).tolist()
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

    """
    if stop is None:
        start, stop = 0, start
    start = numpy.array(start, dtype=int).flatten()
    stop = numpy.array(stop, dtype=int).flatten()
    ordering = ordering.upper()
    start[start < 0] = 0

    indices = _bindex(start, stop, dimensions)

    cross_truncation = cross_truncation*numpy.ones(2)
    lower = cross_truncate(indices, start-1, cross_truncation[0])
    upper = cross_truncate(indices, stop-1, cross_truncation[1])
    indices = indices[lower^upper]

    if "G" in ordering:
        indices = indices[numpy.lexsort([numpy.sum(indices, -1)])]

    if "I" in ordering:
        indices = indices[::-1]

    if "R" in ordering:
        indices = indices[:, ::-1]

    return indices


def _bindex(start, stop, dimensions=1):
    """Backend for the bindex function."""
    # At the beginning the current list of indices just ranges over the
    # last dimension.
    bound = stop.max()
    range_ = numpy.arange(bound, dtype=int)
    indices = range_[:, numpy.newaxis]
    start = start*numpy.ones(dimensions)

    for _ in range(dimensions-1):

        # Repeats the current set of indices.
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = numpy.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices)//bound)[:, numpy.newaxis]

        # Puts them two together.
        indices = numpy.column_stack((front, indices))

    return indices.reshape(-1, dimensions)
