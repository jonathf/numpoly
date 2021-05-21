"""Return an array of zeros with the same shape and type as a given array."""
from __future__ import annotations
from typing import Any, Optional, Sequence

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements

Order = Any
try:
    from typing import Literal, Union
    Order = Union[None, Literal["C"], Literal["F"]]  # type: ignore
except ImportError:
    pass


@implements(numpy.zeros_like)
def zeros_like(
        a: PolyLike,
        dtype: Optional[numpy.typing.DTypeLike] = None,
        order: Order = None,
        subok: bool = True,
        shape: Optional[Sequence[int]] = None,
) -> ndpoly:
    """
    Return an array of zeros with the same shape and type as a given array.

    Args:
        a:
            The shape and data-type of `a` define these same attributes of
            the returned array.
        dtype:
            Overrides the data type of the result.
        order:
            Overrides the memory layout of the result. 'C' means C-order,
            'F' means F-order. If omitted: 'F' if `a` is Fortran contiguous,
            'C' otherwise.
        subok:
            If True, then the newly created array will use the sub-class
            type of 'a', otherwise it will be a base-class array. Defaults
            to True.
        shape:
            Overrides the shape of the result. If order='K' and the number of
            dimensions is unchanged, will try to keep order, otherwise,
            order='C' is implied.

    Returns:
        Array of zeros with the same shape and type as `a`.

    Examples:
        >>> poly = numpoly.monomial(3)
        >>> poly
        polynomial([1, q0, q0**2])
        >>> numpoly.zeros_like(poly)
        polynomial([0, 0, 0])

    """
    del subok
    a = numpoly.aspolynomial(a)
    return numpoly.polynomial(numpy.zeros_like(  # pylint: disable=unexpected-keyword-arg
        a.values[a.keys[0]], dtype=dtype, order=order, shape=shape))
