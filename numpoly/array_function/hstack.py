"""Stack arrays in sequence horizontally (column wise)."""
from __future__ import annotations
from typing import Sequence

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.hstack)
def hstack(tup: Sequence[PolyLike]) -> ndpoly:
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Args:
        tup:
            The arrays must have the same shape along all but the second axis,
            except 1-D arrays which can be any length.

    Returns:
        The array formed by stacking the given arrays.

    Examples:
        >>> poly1 = numpoly.variable(3)
        >>> const1 = numpoly.polynomial([1, 2, 3])
        >>> numpoly.hstack([poly1, const1])
        polynomial([q0, q1, q2, 1, 2, 3])
        >>> const2 = numpoly.polynomial([[1], [2], [3]])
        >>> poly2 = poly1.reshape(3, 1)
        >>> numpoly.hstack([const2, poly2])
        polynomial([[1, q0],
                    [2, q1],
                    [3, q2]])

    """
    arrays = numpoly.align_exponents(*tup)
    coefficients = [numpy.hstack([array.values[key] for array in arrays])
                    for key in arrays[0].keys]
    return numpoly.polynomial_from_attributes(
        exponents=arrays[0].exponents,
        coefficients=coefficients,
        names=arrays[0].names,
        dtype=coefficients[0].dtype,
    )
