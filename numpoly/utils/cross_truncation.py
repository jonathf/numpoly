"""Truncation rules for indices."""
from __future__ import annotations

import numpy
import numpy.typing


def cross_truncate(
        indices: numpy.typing.ArrayLike,
        bound: numpy.typing.ArrayLike,
        norm: float,
) -> numpy.ndarray:
    r"""
    Truncate of indices using L_p norm.

    .. math:
        L_p(x) = (\sum_i |x_i/b_i|^p )^{1/p} \leq 1

    where :math:`b_i` are bounds that each :math:`x_i` should follow.

    Args:
        indices:
            Indices to be truncated.
        bound:
            The bound function for witch the indices can not be larger than.
        norm:
            The `p` in the `L_p`-norm. Support includes both `L_0` and `L_inf`.

    Returns:
        Boolean indices to ``indices`` with True for each index where the
        truncation criteria holds.

    Examples:
        >>> indices = numpy.array(numpy.mgrid[:10, :10]).reshape(2, -1).T
        >>> indices[cross_truncate(indices, 2, norm=0.)].T
        array([[0, 0, 0, 1, 2],
               [0, 1, 2, 0, 0]])
        >>> indices[cross_truncate(indices, 2, norm=1.)].T
        array([[0, 0, 0, 1, 1, 2],
               [0, 1, 2, 0, 1, 0]])
        >>> indices[cross_truncate(indices, [0, 1], norm=1.)].T
        array([[0, 0],
               [0, 1]])

    """
    assert norm >= 0, "negative L_p norm not allowed"
    indices = numpy.asarray(indices)
    bound_ = numpy.broadcast_to(numpy.asfarray(bound).ravel(), (indices.shape[1],))
    nudge_factor = 1e-12*indices.shape[1]

    if numpy.any(bound_ < 0):
        return numpy.zeros((len(indices),), dtype=bool)

    if numpy.any(bound_ == 0):
        out = numpy.all(indices[:, bound_ == 0] == 0, axis=-1)
        if numpy.any(bound_):
            out &= cross_truncate(indices[:, bound_ != 0], bound_[bound_ != 0], norm=norm)
        return out

    if norm == 0:
        out = numpy.sum(indices > 0, axis=-1) <= 1+nudge_factor
        out[numpy.any(indices > bound_, axis=-1)] = False
    elif norm == numpy.inf:
        out = numpy.max(indices/bound_, axis=-1) <= 1+nudge_factor
    else:
        out = numpy.sum((indices/bound_)**norm, axis=-1)**(1./norm) <= 1+nudge_factor

    assert numpy.all(out[numpy.all(indices == 0, axis=-1)])
    return out
