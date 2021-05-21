"""Return a new array of given shape and type, filled with `fill_value`."""
from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.full)
def full(
    shape: Union[int, Sequence[int]],
    fill_value: PolyLike,
    dtype: Optional[numpy.typing.DTypeLike] = None,
    order: str = "C",
) -> ndpoly:
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Args:
        shape:
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        fill_value:
            Fill value. Must be broadcast compatible with `shape`.
        dtype:
            The desired data-type for the array  The default, None, means
            inherit from `fill_value`.
        order:
            Whether to store multidimensional data in C- or Fortran-contiguous
            (row- or column-wise) order in memory. Valid values: "C", "F".

    Returns:
        Array of `fill_value` with the given shape, dtype, and order.

    Examples:
        >>> numpoly.full((2, 4), 4)
        polynomial([[4, 4, 4, 4],
                    [4, 4, 4, 4]])
        >>> q0 = numpoly.variable()
        >>> numpoly.full(3, q0**2-1)
        polynomial([q0**2-1, q0**2-1, q0**2-1])

    """
    fill_value = numpoly.aspolynomial(fill_value)
    if dtype is None:
        dtype = fill_value.dtype
    shape = tuple((shape,) if isinstance(shape, int) else shape)
    out = numpoly.ndpoly(
        exponents=fill_value.exponents,
        shape=shape,
        names=fill_value.indeterminants,
        dtype=dtype,
        order=order,
    )
    for key in fill_value.keys:
        out.values[key] = fill_value.values[key]
    return out
