"""Join a sequence of arrays along a new axis."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.stack)
def stack(
    arrays: Sequence[PolyLike],
    axis: int = 0,
    out: Optional[ndpoly] = None,
) -> ndpoly:
    """
    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    Args:
        arrays:
            Each array must have the same shape.
        axis:
            The axis in the result array along which the input arrays are
            stacked.
        out:
            If provided, the destination to place the result. The shape must be
            correct, matching that of what stack would have returned if no out
            argument were specified.

    Returns:
        The stacked array has one more dimension than the input arrays.

    Examples:
        >>> poly = numpoly.variable(3)
        >>> const = numpoly.polynomial([1, 2, 3])
        >>> numpoly.stack([poly, const])
        polynomial([[q0, q1, q2],
                    [1, 2, 3]])
        >>> numpoly.stack([poly, const], axis=-1)
        polynomial([[q0, 1],
                    [q1, 2],
                    [q2, 3]])

    """
    arrays = numpoly.align_exponents(*arrays)
    if out is None:
        coefficients = [numpy.stack(
            [array.values[key] for array in arrays], axis=axis)
                        for key in arrays[0].keys]
        out = numpoly.polynomial_from_attributes(
            exponents=arrays[0].exponents,
            coefficients=coefficients,
            names=arrays[0].names,
            dtype=coefficients[0].dtype,
        )
    else:
        for key in out.keys:
            if key in arrays[0].keys:
                numpy.stack([array.values[key] for array in arrays],
                            out=out.values[key], axis=axis)
    return out
