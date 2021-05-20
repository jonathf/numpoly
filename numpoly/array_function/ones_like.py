"""Return an array of ones with the same shape and type as a given array."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.ones_like)
def ones_like(
    a: PolyLike,
    dtype: Optional[numpy.typing.DTypeLike] = None,
    order: str = 'K',
    subok: bool = True,
    shape: Optional[Sequence[int]] = None,
) -> ndpoly:
    """
    Return an array of ones with the same shape and type as a given array.

    Args:
        a:
            The shape and data-type of `a` define these same attributes of
            the returned array.
        dtype:
            Overrides the data type of the result.
        order:
            Overrides the memory layout of the result. 'C' means C-order,
            'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
            'C' otherwise. 'K' means match the layout of `a` as closely
            as possible.
        subok:
            If True, then the newly created array will use the sub-class
            type of 'a', otherwise it will be a base-class array. Defaults
            to True.
        shape:
            Overrides the shape of the result. If order='K' and the number of
            dimensions is unchanged, will try to keep order, otherwise,
            order='C' is implied.

    Returns:
        Array of ones with the same shape and type as `a`.

    Examples:
        >>> poly = numpoly.monomial(3)
        >>> poly
        polynomial([1, q0, q0**2])
        >>> numpoly.ones_like(poly)
        polynomial([1, 1, 1])

    """
    del subok
    if not isinstance(a, numpy.ndarray):
        a = numpoly.polynomial(a)
    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    if order in ("A", "K"):
        order = "F" if a.flags["F_CONTIGUOUS"] else "C"
    return numpoly.polynomial(numpy.ones(shape, dtype=dtype, order=order))
