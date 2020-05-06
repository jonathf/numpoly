"""Truncation rules for indices."""
import numpy


def cross_truncate(indices, bound, norm):
    r"""
    Truncate of indices using L_p norm.

    .. math:
        L_p(x) = \sum_i |x_i/b_i|^p ^{1/p} \leq 1

    where :math:`b_i` are bounds that each :math:`x_i` should follow.

    Args:
        indices (Sequence[int]):
            Indices to be truncated.
        bound (int, Sequence[int]):
            The bound function for witch the indices can not be larger than.
        norm (float, Sequence[float]):
            The `p` in the `L_p`-norm. Support includes both `L_0` and `L_inf`.

    Returns:
        Boolean indices to ``indices`` with True for each index where the
        truncation criteria holds.

    Examples:
        >>> indices = numpy.array(numpy.mgrid[:10, :10]).reshape(2, -1).T
        >>> indices[cross_truncate(indices, 2, norm=0)].T
        array([[0, 0, 0, 1, 2],
               [0, 1, 2, 0, 0]])
        >>> indices[cross_truncate(indices, 2, norm=1)].T
        array([[0, 0, 0, 1, 1, 2],
               [0, 1, 2, 0, 1, 0]])
        >>> indices[cross_truncate(indices, 2, norm=numpy.inf)].T
        array([[0, 0, 0, 1, 1, 1, 2, 2, 2],
               [0, 1, 2, 0, 1, 2, 0, 1, 2]])

    """
    assert norm >= 0, "negative L_p norm not allowed"
    bound = numpy.asfarray(bound)
    if numpy.any(bound < 0):
        out = numpy.zeros((len(indices),), dtype=bool)
    elif norm == 0:
        out = numpy.sum(indices > 0, axis=-1) <= 1
        out[numpy.any(indices > bound, axis=-1)] = False
    elif norm == numpy.inf:
        out = numpy.max(indices/bound, axis=-1) <= 1
    else:
        out = numpy.sum((indices/bound)**norm, axis=-1)**(1./norm) <= 1
    if not numpy.any(bound < 0):
        out[numpy.all(indices == 0, axis=-1)] = True
    return out
