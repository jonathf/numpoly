"""Sort keys lexicographically."""
import numpy


def bsort(values, ordering="G"):
    """
    Sort keys lexicographically.

    Same as ``numpy.lexsort``, but also support inverse graded reverse
    lexicographical ordering.

    Args:
        values (Sequence(Sequence[int])):
            Values to sort.
        ordering (str):
            Short hand for the criteria to sort the indices by.

            ``G``
                Graded sorting, meaning the indices are always sorted by the
                index sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will
                therefore be consider larger than both ``(3, 1, 1)`` and
                ``(1, 1, 3)``.
            ``I``
                Reversed, meaning the biggest values are in the front instead
                of the back.
            ``R``
                Inverse lexicographical sorting meaning that ``(1, 3)`` is
                considered bigger than ``(3, 1)``, instead of the opposite.

    Returns:
        (numpy.ndarray):
            Array of indices that sort the keys along the specified axis.

    Examples:
        >>> indices = numpy.array([[0, 0, 0, 1, 2, 1],
        ...                        [1, 2, 0, 0, 0, 1]])
        >>> indices[:, numpy.lexsort(indices)]
        array([[0, 1, 2, 0, 1, 0],
               [0, 0, 0, 1, 1, 2]])
        >>> indices[:, numpoly.bsort(indices, ordering="I")]
        array([[2, 1, 1, 0, 0, 0],
               [0, 1, 0, 2, 1, 0]])
        >>> indices[:, numpoly.bsort(indices, ordering="")]
        array([[0, 0, 0, 1, 1, 2],
               [0, 1, 2, 0, 1, 0]])
        >>> indices[:, numpoly.bsort(indices, ordering="G")]
        array([[0, 0, 1, 0, 1, 2],
               [0, 1, 0, 2, 1, 0]])
        >>> indices[:, numpoly.bsort(indices, ordering="GRI")]
        array([[0, 1, 2, 0, 1, 0],
               [2, 1, 0, 1, 0, 0]])
        >>> indices = numpy.array([4, 5, 6, 3, 2, 1])
        >>> indices[numpoly.bsort(indices)]
        array([1, 2, 3, 4, 5, 6])

    """
    ordering = ordering.upper()
    values = numpy.atleast_2d(values)

    if "R" not in ordering:
        values = values[::-1]

    indices = numpy.array(numpy.lexsort(values))

    if "G" in ordering:
        indices = indices[numpy.argsort(
            numpy.sum(values[:, indices], axis=0))].T

    if "I" in ordering:
        indices = indices[::-1]

    return indices
