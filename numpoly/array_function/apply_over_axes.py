"""Apply a function repeatedly over multiple axes."""
from __future__ import annotations
from functools import wraps
from typing import Callable

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.apply_over_axes)
def apply_over_axes(
    func: Callable,
    a: PolyLike,
    axes: numpy.typing.ArrayLike,
) -> ndpoly:
    """
    Apply a function repeatedly over multiple axes.

    `func` is called as `res = func(a, axis)`, where `axis` is the first
    element of `axes`.  The result `res` of the function call must have either
    the same dimensions as `a` or one less dimension.  If `res` has one less
    dimension than `a`, a dimension is inserted before `axis`.  The call to
    `func` is then repeated for each axis in `axes`, with `res` as the first
    argument.

    Args:
        func:
            This function must take two arguments, `func(a, axis)`.
        a:
            Input array.
        axes:
            Axes over which `func` is applied; the elements must be integers.

    Returns:
        The output array.  The number of dimensions is the same as `a`, but
        the shape can be different.  This depends on whether `func` changes
        the shape of its output with respect to its input.

    Examples:
        >>> polynomial = numpy.arange(24).reshape(2, 4, 3)
        >>> polynomial = polynomial*numpoly.variable(3)
        >>> numpoly.apply_over_axes(
        ...     func=numpoly.sum, a=polynomial, axes=1)
        polynomial([[[18*q0, 22*q1, 26*q2]],
        <BLANKLINE>
                    [[66*q0, 70*q1, 74*q2]]])
        >>> numpoly.apply_over_axes(
        ...     func=numpoly.sum, a=polynomial, axes=[0, 2])
        polynomial([[[16*q2+14*q1+12*q0],
                     [22*q2+20*q1+18*q0],
                     [28*q2+26*q1+24*q0],
                     [34*q2+32*q1+30*q0]]])

    """
    @wraps(func)
    def wrapper_func(array, axis):
        """Wrap func function."""
        # Align indeterminants in case slicing changed them
        array = numpoly.polynomial(array, names=a.indeterminants)
        array, _ = numpoly.align.align_indeterminants(array, a.indeterminants)
        # Evaluate function
        out = func(array, axis=axis)
        # Restore indeterminants in case func changed them.
        out, _ = numpoly.align.align_indeterminants(out, a.indeterminants)
        return out

    # Initiate wrapper
    a = numpoly.aspolynomial(a)
    out = numpy.apply_over_axes(wrapper_func, a=a.values, axes=axes)
    return out
