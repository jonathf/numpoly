"""Numerical positive, element-wise."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements, simple_dispatch


@implements(numpy.positive)  # pylint: disable=no-member
def positive(
    x: PolyLike,
    out: Optional[ndpoly] = None,
    where: numpy.typing.ArrayLike = True,
    **kwargs: Any,
) -> ndpoly:
    """
    Numerical positive, element-wise.

    Args:
        x:
            Input array.
        out:
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where : array_like, optional
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        Returned array or scalar: `y = +x`.
        This is a scalar if `x` is a scalar.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.positive([-0, 0, -q0, q0])
        polynomial([0, 0, -q0, q0])

    """
    return simple_dispatch(
        numpy_func=numpy.positive,  # pylint: disable=no-member
        inputs=(x,),
        out=None if out is None else (out,),
        where=where,
        **kwargs
    )
