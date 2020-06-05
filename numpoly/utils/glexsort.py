"""Sort keys lexicographically."""
import numpy


def glexsort(keys, graded=False, reverse=False):
    """
    Sort keys graded lexicographically.

    Same as ``numpy.lexsort``, but also support graded reverse
    lexicographical ordering.

    Args:
        keys (Sequence(Sequence[int])):
            Values to sort.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will therefore be
            consider larger than both ``(3, 1, 1)`` and ``(1, 1, 3)``.
        reverse (bool):
            Reverse lexicographical sorting meaning that ``(1, 3)`` is
            considered smaller than ``(3, 1)``, instead of the opposite.

    Returns:
        (numpy.ndarray):
            Array of indices that sort the keys along the specified axis.

    Examples:
        >>> indices = numpy.array([[0, 0, 0, 1, 2, 1],
        ...                        [1, 2, 0, 0, 0, 1]])
        >>> indices[:, numpy.lexsort(indices)]
        array([[0, 1, 2, 0, 1, 0],
               [0, 0, 0, 1, 1, 2]])
        >>> indices[:, numpoly.glexsort(indices)]
        array([[0, 1, 2, 0, 1, 0],
               [0, 0, 0, 1, 1, 2]])
        >>> indices[:, numpoly.glexsort(indices, reverse=True)]
        array([[0, 0, 0, 1, 1, 2],
               [0, 1, 2, 0, 1, 0]])
        >>> indices[:, numpoly.glexsort(indices, graded=True)]
        array([[0, 1, 0, 2, 1, 0],
               [0, 0, 1, 0, 1, 2]])
        >>> indices[:, numpoly.glexsort(indices, graded=True, reverse=True)]
        array([[0, 0, 1, 0, 1, 2],
               [0, 1, 0, 2, 1, 0]])
        >>> indices = numpy.array([4, 5, 6, 3, 2, 1])
        >>> indices[numpoly.glexsort(indices)]
        array([1, 2, 3, 4, 5, 6])

    """
    keys = numpy.atleast_2d(keys)

    if reverse:
        keys = keys[::-1]

    indices = numpy.array(numpy.lexsort(keys))
    if graded:
        indices = indices[numpy.argsort(
            numpy.sum(keys[:, indices], axis=0))].T
    return indices
