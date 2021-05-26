"""Join a sequence of arrays along an existing axis."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.concatenate)
def concatenate(
    arrays: Sequence[PolyLike],
    axis: int = 0,
    out: Optional[ndpoly] = None,
) -> ndpoly:
    """
    Join a sequence of arrays along an existing axis.

    Args:
        arrays:
            The arrays must have the same shape, except in the dimension
            corresponding to `axis` (the first, by default).
        axis:
            The axis along which the arrays will be joined.  If axis is None,
            arrays are flattened before use.  Default is 0.
        out:
            If provided, the destination to place the result. The shape must be
            correct, matching that of what concatenate would have returned if
            no out argument were specified.

    Returns:
        The concatenated array.

    Examples:
        >>> const = numpy.array([[1, 2], [3, 4]])
        >>> poly = numpoly.variable(2).reshape(1, 2)
        >>> numpoly.concatenate((const, poly), axis=0)
        polynomial([[1, 2],
                    [3, 4],
                    [q0, q1]])
        >>> numpoly.concatenate((const, poly.T), axis=1)
        polynomial([[1, 2, q0],
                    [3, 4, q1]])
        >>> numpoly.concatenate((const, poly), axis=None)
        polynomial([1, 2, 3, 4, q0, q1])

    """
    arrays = numpoly.align_exponents(*arrays)
    if out is None:
        coefficients = [numpy.concatenate(
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
                numpy.concatenate([array.values[key] for array in arrays],
                                  out=out.values[key], axis=axis)
    return out
