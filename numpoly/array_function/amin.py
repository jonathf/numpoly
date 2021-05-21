"""Return the minimum of an array or minimum along an axis."""
from __future__ import annotations
from typing import Any, Optional, Sequence, Union

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.amin)
def amin(
    a: PolyLike,
    axis: Union[None, int, Sequence[int]] = None,
    out: Optional[ndpoly] = None,
    **kwargs: Any,
) -> ndpoly:
    """
    Return the minimum of an array or minimum along an axis.

    Args:
        a:
            Input data.
        axis:
            Axis or axes along which to operate.  By default, flattened input
            is used. If this is a tuple of ints, the minimum is selected over
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

    Returns:
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension ``a.ndim-1``.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.amin([13, 7])
        polynomial(7)
        >>> numpoly.amin([1, q0, q0**2, q1])
        polynomial(1)
        >>> numpoly.amin([q0, q1, q0**2])
        polynomial(q0)
        >>> numpoly.amin([[3*q0**2, q0**2],
        ...               [2*q0**2, 4*q0**2]], axis=1)
        polynomial([q0**2, 2*q0**2])

    """
    del out
    poly = numpoly.aspolynomial(a)
    options = numpoly.get_options()
    proxy = numpoly.sortable_proxy(
        poly, graded=options["sort_graded"], reverse=options["sort_reverse"])
    indices = numpy.amin(proxy, axis=axis, **kwargs)
    out = poly[numpy.isin(proxy, indices)]
    out = out[numpy.argsort(indices.ravel())]
    return numpoly.reshape(out, indices.shape)
