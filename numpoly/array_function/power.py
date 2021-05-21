"""First array elements raised to powers from second array, element-wise."""
from __future__ import annotations
from typing import Any

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.power)
def power(x1: PolyLike, x2: PolyLike, **kwargs: Any) -> ndpoly:
    """
    First array elements raised to powers from second array, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in
    `x2`. `x1` and `x2` must be broadcastable to the same shape. Note that an
    integer type raised to a negative integer power will raise a ValueError.

    Args:
        x1:
            The bases.
        x2:
            The exponents. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the
            output).
        out:
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where:
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        The bases in `x1` raised to the exponents in `x2`.
        This is a scalar if both `x1` and `x2` are scalars.

    Examples:
        >>> q0 = numpoly.variable()
        >>> (q0+1)**3
        polynomial(q0**3+3*q0**2+3*q0+1)
        >>> q0q1 = numpoly.variable(2)
        >>> (q0q1+1)**[1, 2]
        polynomial([q0+1, q1**2+2*q1+1])

    """
    x1 = numpoly.aspolynomial(x1)
    x2 = numpoly.aspolynomial(x2).tonumpy().astype(int)

    if not x2.shape:
        out = numpoly.ndpoly.from_attributes(
            [(0,)], [numpy.ones(x1.shape, dtype=x1._dtype)], x1.names[:1])
        for _ in range(x2.item()):
            out = numpoly.multiply(out, x1, **kwargs)

    elif x1.shape:
        if x2.shape[-1] == 1:
            if x1.shape[-1] == 1:
                out = numpoly.power(x1.T[0].T, x2.T[0].T).T[numpy.newaxis].T
            else:
                out = numpoly.concatenate([power(x, x2.T[0])[numpy.newaxis]
                                           for x in x1.T], axis=0).T
        elif x1.shape[-1] == 1:
            out = numpoly.concatenate([power(x1.T[0].T, x.T).T[numpy.newaxis]
                                       for x in x2.T], axis=0).T
        else:
            out = numpoly.concatenate([power(x1_, x2_).T[numpy.newaxis]
                                       for x1_, x2_ in zip(x1.T, x2.T)], axis=0).T
    else:
        out = numpoly.concatenate([power(x1, x.T).T[numpy.newaxis]
                                   for x in x2.T], axis=0).T
    return numpoly.polynomial(out)
