"""Counts the number of non-zero values in the array a."""
from __future__ import annotations
from typing import Any, Sequence, Union

import numpy
import numpy.typing
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.count_nonzero)
def count_nonzero(
    x: PolyLike,
    axis: Union[None, int, Sequence[int]] = None,
    **kwargs: Any,
) -> Union[int, numpy.ndarray]:
    """
    Count the number of non-zero values in the array a.

    Args:
        x:
            The array for which to count non-zeros.
        axis: (Union[int, Tuple[int], None]):
            Axis or tuple of axes along which to count non-zeros. Default is
            None, meaning that non-zeros will be counted along a flattened
            version of a.

    Returns:
        Number of non-zero values in the array along a given axis. Otherwise,
        the total number of non-zero values in the array is returned.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.count_nonzero([q0])
        1
        >>> numpoly.count_nonzero([[0, q0*q0, 0, 0],
        ...                        [q0+1, 0, 2*q0, 7*q0]])
        4
        >>> numpoly.count_nonzero([[0, q0+q1, 0, 0],
        ...                        [3*q1, 0, 2, 7*q0]], axis=0)
        array([1, 1, 1, 1])
        >>> numpoly.count_nonzero([[0, q1, 0, 0],
        ...                        [q0, 0, 2*q0, 6*q1]], axis=1)
        array([1, 3])

    """
    a = numpoly.aspolynomial(x)
    index = numpy.any(numpy.asarray(a.coefficients), axis=0)
    return numpy.count_nonzero(index, axis=axis)
