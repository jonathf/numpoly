"""Compute the determinant of an polynomial array."""
from __future__ import annotations

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.linalg.det)
def det(a: PolyLike) -> ndpoly:
    """
    Compute the determinant of an polynomial array.

    Args:
        a:
            Input array to compute determinants for.
            Shape on form `(..., M, M)`.

    Returns:
        Determinant of `a`. Shape `(...)`.

    Notes:
        Broadcasting rules apply, see the `numpy.linalg` documentation for
        details.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> a = numpoly.polynomial([[[1, 2], [3, 4]], [[1, q0], [q1, 1]]])
        >>> numpoly.det(a)
        polynomial([-2, -q0*q1+1])

    """
    a = numpoly.aspolynomial(a)
    assert a.ndim >= 2, a
    assert a.shape[-2] == a.shape[-1], a.shape
    dims = a.shape[-1]
    index = (slice(None),)*(a.ndim-2)
    if dims == 2:
        return a[index+(0, 0)]*a[index+(1, 1)]-a[index+(1, 0)]*a[index+(0, 1)]
    out = numpoly.zeros_like(a, shape=a.shape[:-2])
    r = numpy.arange(1, dims, dtype=int)
    for idx in range(dims):
        idx0 = index+(0, idx)
        idx1 = index+(slice(1, None), (r+idx)%dims)
        out = out+a[idx0]*det(a[idx1])
    return out
