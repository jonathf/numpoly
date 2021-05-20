"""Test whether any array element along a given axis evaluates to True."""
from __future__ import annotations
from typing import Any, Optional, Sequence, Union

import numpy
import numpy.typing
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.any)
def any(
    a: PolyLike,
    axis: Union[None, int, Sequence[int]] = None,
    out: Optional[numpy.ndarray] = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> Optional[numpy.ndarray]:
    """
    Test whether any array element along a given axis evaluates to True.

    Returns single boolean unless `axis` is not ``None``

    Args:
        a:
            Input array or object that can be converted to an array.
        axis:
            Axis or axes along which a logical OR reduction is performed. The
            default (`axis` = `None`) is to perform a logical OR over all the
            dimensions of the input array. `axis` may be negative, in which
            case it counts from the last to the first axis. If this is a tuple
            of ints, a reduction is performed on multiple axes, instead of
            a single axis or all the axes as before.
        out:
            Alternate output array in which to place the result.  It must have
            the same shape as the expected output and its type is preserved
            (e.g., if it is of type float, then it will remain so, returning
            1.0 for True and 0.0 for False, regardless of the type of `a`).
        keepdims:
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.

    Returns:
        A new boolean or `ndarray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.any(q0)
        True
        >>> numpoly.any(0*q0)
        False
        >>> numpoly.any([1, q0, 0])
        True
        >>> numpoly.any([[True*q0, False], [True, True]], axis=0)
        array([ True,  True])

    """
    a = numpoly.aspolynomial(a)
    coefficients = numpy.any(numpy.asarray(a.coefficients), axis=0)
    index = numpy.asarray(coefficients, dtype=bool)
    return numpy.any(index, axis=axis, out=out, keepdims=keepdims)
