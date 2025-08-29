"""Return the cumulative sum of the elements along a given axis."""

from __future__ import annotations
from typing import Optional

import numpy
import numpy.typing

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements, simple_dispatch


@implements(numpy.cumsum)
def cumsum(
    a: PolyLike,
    axis: Optional[int] = None,
    dtype: Optional[numpy.typing.DTypeLike] = None,
    out: Optional[ndpoly] = None,
) -> ndpoly:
    """
    Return the cumulative sum of the elements along a given axis.

    Args:
        a:
            Input array.
        axis:
            Axis along which the cumulative sum is computed. The default
            (None) is to compute the cumsum over the flattened array.
        dtype:
            Type of the returned array and of the accumulator in which the
            elements are summed.  If `dtype` is not specified, it defaults
            to the dtype of `a`, unless `a` has an integer dtype with a
            precision less than that of the default platform integer.  In
            that case, the default platform integer is used.
        out:
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary.

    Args:
        A new array holding the result is returned unless `out` is specified,
        in which case a reference to `out` is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None or `a` is
        a 1-d array.

    Example:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[1, q0, 3], [4, 5, q1]])
        >>> poly
        polynomial([[1, q0, 3],
                    [4, 5, q1]])
        >>> numpoly.cumsum(poly)
        polynomial([1, q0+1, q0+4, q0+8, q0+13, q1+q0+13])
        >>> numpoly.cumsum(poly, dtype=float, axis=1)
        polynomial([[1.0, q0+1.0, q0+4.0],
                    [4.0, 9.0, q1+9.0]])
        >>> numpoly.cumsum(poly, axis=0)
        polynomial([[1, q0, 3],
                    [5, q0+5, q1+3]])

    """
    return simple_dispatch(
        numpy_func=numpy.cumsum,
        inputs=(a,),
        out=None if out is None else (out,),
        axis=axis,
        dtype=dtype,
    )
