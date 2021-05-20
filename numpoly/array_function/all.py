"""Test whether all array elements along a given axis evaluate to True."""
from __future__ import annotations
from typing import Any, Optional, Sequence, Union

import numpy
import numpy.typing
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.all)
def all(
    a: PolyLike,
    axis: Union[None, int, Sequence[int]] = None,
    out: Optional[numpy.ndarray] = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> Optional[numpy.ndarray]:
    """
    Test whether all array elements along a given axis evaluate to True.

    Args:
        a:
            Input array or object that can be converted to an array.
        axis:
            Axis or axes along which a logical AND reduction is performed. The
            default (`axis` = `None`) is to perform a logical AND over all the
            dimensions of the input array. `axis` may be negative, in which
            case it counts from the last to the first axis. If this is a tuple
            of ints, a reduction is performed on multiple axes, instead of
            a single axis or all the axes as before.
        out:
            Alternate output array in which to place the result. It must have
            the same shape as the expected output and its type is preserved
            (e.g., if ``dtype(out)`` is float, the result will consist of 0.0's
            and 1.0's).
        keepdims:
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

    Returns:
        A new boolean or array is returned unless `out` is specified, in which
        case a reference to `out` is returned.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.all(q0)
        True
        >>> numpoly.all(0*q0)
        False
        >>> numpoly.all([1, q0, 0])
        False
        >>> numpoly.all([[True*q0, False], [True, True]], axis=0)
        array([ True, False])

    """
    a = numpoly.aspolynomial(a)
    coefficients = numpy.any(numpy.asarray(a.coefficients), axis=0)
    index = numpy.asarray(coefficients, dtype=bool)
    return numpy.all(index, axis=axis, out=out, keepdims=keepdims)
