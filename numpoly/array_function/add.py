"""Add arguments element-wise."""

from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements, simple_dispatch


@implements(numpy.add)
def add(
    x1: PolyLike,
    x2: PolyLike,
    out: Optional[ndpoly] = None,
    where: numpy.typing.ArrayLike = True,
    **kwargs: Any,
) -> ndpoly:
    """
    Add arguments element-wise.

    Args:
        x1, x2:
            The arrays to be added. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the
            output).
        out (Optional[numpy.ndarray]) -> ndpoly:
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (Union[bool, numpy.ndarray]) -> ndpoly:
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Return:
        The sum of `x1` and `x2`, element-wise. This is a scalar if both
        `x1` and `x2` are scalars.

    Example:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.add(q0, 4)
        polynomial(q0+4)
        >>> poly1 = q0**numpy.arange(9).reshape((3, 3))
        >>> poly2 = q1**numpy.arange(3)
        >>> numpoly.add(poly1, poly2)
        polynomial([[2, q1+q0, q1**2+q0**2],
                    [q0**3+1, q0**4+q1, q0**5+q1**2],
                    [q0**6+1, q0**7+q1, q0**8+q1**2]])

    """
    return simple_dispatch(
        numpy_func=numpy.add,
        inputs=(x1, x2),
        out=None if out is None else (out,),
        where=where,
        **kwargs,
    )
