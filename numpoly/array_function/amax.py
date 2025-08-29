"""Return the maximum of an array or maximum along an axis."""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements

if tuple(int(v) for v in numpy.__version__.split(".")) >= (1, 25, 0):
    impls = [numpy.amax, numpy.max, max]
else:
    impls = [numpy.amax, max]


@implements(*impls)
def amax(
    a: PolyLike,
    axis: Union[None, int, Sequence[int]] = None,
    out: Optional[ndpoly] = None,
    **kwargs: Any,
) -> ndpoly:
    """
    Return the maximum of an array or maximum along an axis.

    Args:
        a:
            Input data.
        axis:
            Axis or axes along which to operate.  By default, flattened input
            is used. If this is a tuple of ints, the maximum is selected over
            multiple axes, instead of a single axis or all the axes as before.
        out:
            Alternative output array in which to place the result. Must be of
            the same shape and buffer length as the expected output.

        keepdims:
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.

            If the default value is passed, then `keepdims` will not be passed
            through to the `amax` method of sub-classes of `ndarray`, however
            any non-default value will be.  If the sub-class' method does not
            implement `keepdims` any exceptions will be raised.
        initial:
            The minimum value of an output element. Must be present to allow
            computation on empty slice.
        where:
            Elements to compare for the maximum.

    Return:
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension ``a.ndim-1``.

    Example:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.amax([13, 7])
        polynomial(13)
        >>> numpoly.amax([1, q0, q0**2, q1])
        polynomial(q0**2)
        >>> numpoly.amax([1, q0, q1])
        polynomial(q1)
        >>> numpoly.amax([[3*q0**2, 5*q0**2],
        ...               [2*q0**2, 4*q0**2]], axis=0)
        polynomial([3*q0**2, 5*q0**2])

    """
    del out
    a = numpoly.aspolynomial(a)
    options = numpoly.get_options()
    proxy = numpoly.sortable_proxy(
        a, graded=options["sort_graded"], reverse=options["sort_reverse"]
    )
    indices = numpy.amax(proxy, axis=axis, **kwargs)
    out = a[numpy.isin(proxy, indices)]
    out = out[numpy.argsort(indices.ravel())]
    return numpoly.reshape(out, indices.shape)
