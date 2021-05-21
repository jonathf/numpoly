"""Return a new array of given shape and type, filled with ones."""
from __future__ import annotations
from typing import Any, Sequence

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly
from ..dispatch import implements

Order = Any
try:
    from typing import Literal, Union
    Order = Union[Literal["C"], Literal["F"], None]  # type: ignore
except ImportError:
    pass


@implements(numpy.ones)
def ones(
    shape: Union[int, Sequence[int]],
    dtype: numpy.typing.DTypeLike = float,
    order: Order = "C",
) -> ndpoly:
    """
    Return a new array of given shape and type, filled with ones.

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
        Array of ones with the given shape, dtype, and order.

    Examples:
        >>> numpoly.ones(5)
        polynomial([1.0, 1.0, 1.0, 1.0, 1.0])

    """
    return numpoly.polynomial(numpy.ones(shape, dtype=dtype, order=order))
