"""Return the indices of the elements that are non-zero."""
from __future__ import annotations
from typing import Any, Tuple

import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.nonzero)
def nonzero(x: PolyLike, **kwargs: Any) -> Tuple[numpy.ndarray, ...]:
    """
    Return the indices of the elements that are non-zero.

    Args:
        x:
            Input array.

    Returns:
        Indices of elements that are non-zero.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[3*q0, 0, 0],
        ...                            [0, 4*q1, 0],
        ...                            [5*q0+q1, 6*q0, 0]])
        >>> poly
        polynomial([[3*q0, 0, 0],
                    [0, 4*q1, 0],
                    [q1+5*q0, 6*q0, 0]])
        >>> numpoly.nonzero(poly)
        (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
        >>> poly[numpoly.nonzero(poly)]
        polynomial([3*q0, 4*q1, q1+5*q0, 6*q0])

    """
    x = numpoly.aspolynomial(x)
    return numpy.nonzero(numpy.any(numpy.asarray(x.coefficients), axis=0))
