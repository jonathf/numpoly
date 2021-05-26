"""Calculate the n-th discrete difference along the given axis."""
from __future__ import annotations
from typing import Optional

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.diff)
def diff(
    a: PolyLike,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[PolyLike] = None,
    append: Optional[PolyLike] = None,
) -> ndpoly:
    """
    Calculate the n-th discrete difference along the given axis.

    The first difference is given by ``out[i] = a[i+1] - a[i]`` along the given
    axis, higher differences are calculated by using `diff` recursively.

    Args:
        a:
            Input array
        n:
            The number of times values are "differenced". If zero, the input is
            returned as-is.
        axis:
            The axis along which the difference is taken, default is the last
            axis.
        prepend, append:
            Values to prepend or append to `a` along axis prior to
            performing the difference. Scalar values are expanded to
            arrays with length 1 in the direction of axis and the shape
            of the input array in along all other axes.  Otherwise the
            dimension and shape must match `a` except along axis.

    Returns:
        The n-th differences. The shape of the output is the same as `a` except
        along `axis` where the dimension is smaller by `n`. The type of the
        output is the same as the type of the difference between any two
        elements of `a`. This is the same as the type of `a` in most cases.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([1, q0, q1, q0**2, q1-1])
        >>> numpoly.diff(poly)
        polynomial([q0-1, q1-q0, q0**2-q1, -q0**2+q1-1])
        >>> numpoly.diff(poly, n=2)
        polynomial([q1-2*q0+1, q0**2-2*q1+q0, -2*q0**2+2*q1-1])
        >>> poly = numpoly.polynomial([[q0, 1], [2, q1]])
        >>> numpoly.diff(poly)
        polynomial([[-q0+1],
                    [q1-2]])
        >>> numpoly.diff(poly, prepend=7, append=q1)
        polynomial([[q0-7, -q0+1, q1-1],
                    [-5, q1-2, 0]])

    """
    if append is not None:
        if prepend is not None:
            a, append, prepend = numpoly.align_exponents(a, append, prepend)
        else:
            a, append = numpoly.align_exponents(a, append)
    elif prepend is not None:
        a, prepend = numpoly.align_exponents(a, prepend)
    else:
        a = numpoly.aspolynomial(a)

    out = None
    for key in a.keys:

        kwargs = {}
        if append is not None:
            kwargs["append"] = append.values[key]
        if prepend is not None:
            kwargs["prepend"] = prepend.values[key]
        tmp = numpy.diff(a.values[key], n=n, axis=axis, **kwargs)

        if out is None:
            out = numpoly.ndpoly(
                exponents=a.exponents,
                shape=tmp.shape,
                names=a.indeterminants,
                dtype=tmp.dtype,
            )
        out.values[key] = tmp
    assert out is not None
    return numpoly.clean_attributes(out)
