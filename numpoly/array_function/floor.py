"""Return the floor of the input, element-wise."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements, simple_dispatch


@implements(numpy.floor)
def floor(
    x: PolyLike,
    out: Optional[ndpoly] = None,
    where: numpy.typing.ArrayLike = True,
    **kwargs: Any,
) -> ndpoly:
    r"""
    Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Args:
        x:
            Input data.
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
        The floor of each element in `x`. This is a scalar if `x` is a scalar.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.floor([-1.7*q0, q0-1.5, -0.2,
        ...                3.2+1.5*q0, 1.7, 2.0])
        polynomial([-2.0*q0, q0-2.0, -1.0, q0+3.0, 1.0, 2.0])

    """
    return simple_dispatch(
        numpy_func=numpy.floor,
        inputs=(x,),
        out=None if out is None else (out,),
        where=where,
        **kwargs
    )
