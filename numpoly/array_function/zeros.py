"""Return a new array of given shape and type, filled with zeros."""
from __future__ import annotations
from typing import Sequence, Union

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly
from ..dispatch import implements


@implements(numpy.zeros)
def zeros(
    shape: Union[int, Sequence[int]],
    dtype: numpy.typing.DTypeLike = float,
    order: str = "C",
) -> ndpoly:
    """
    Return a new array of given shape and type, filled with zeros.

    Args:
        shape:
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype:
            The desired data-type for the array, e.g., `numpy.int8`.
            Default is `numpy.float64`.
        order:
            Whether to store multi-dimensional data in row-major
            (C-style) or column-major (Fortran-style) order in
            memory.

    Returns:
        Array of zeros with the given shape, dtype, and order.

    Examples:
        >>> numpoly.zeros(5)
        polynomial([0.0, 0.0, 0.0, 0.0, 0.0])

    """
    return numpoly.polynomial(numpy.zeros(shape, dtype=dtype, order=order))
