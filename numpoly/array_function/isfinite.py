"""Test element-wise for finiteness (not infinity or not Not a Number)."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpy.typing

from ..baseclass import PolyLike
from ..dispatch import implements, simple_dispatch


@implements(numpy.isfinite)
def isfinite(
        x: PolyLike,
        out: Optional[numpy.ndarray] = None,
        where: numpy.typing.ArrayLike = True,
        **kwargs: Any,
) -> numpy.ndarray:
    """
    Test element-wise for finiteness (not infinity or not Not a Number).

    The result is returned as a boolean array.

    Args:
        x:
            Input values.
        out:
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where:
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value.
            Note that if an uninitialized `out` array is created via the
            default ``out=None``, locations within it where the condition is
            False will remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        True where ``x`` is not positive infinity, negative infinity, or NaN;
        false otherwise. This is a scalar if `x` is a scalar.

    Notes:
        Not a Number, positive infinity and negative infinity are considered to
        be non-finite.

    Examples:
        >>> numpoly.isfinite(1)
        True
        >>> numpoly.isfinite(0)
        True
        >>> numpoly.isfinite(numpy.nan*numpoly.variable())
        False
        >>> numpoly.isfinite(numpy.inf)
        False
        >>> numpoly.isfinite(numpy.NINF)
        False
        >>> numpoly.isfinite([numpy.log(-1.), 1., numpy.log(0)])
        array([False,  True, False])

    """
    out_ = simple_dispatch(
        numpy_func=numpy.isfinite,
        inputs=(x,),
        where=where,
        **kwargs
    )
    if out is None:
        out_ = numpy.all(numpy.asarray(out_.coefficients), axis=0)
    else:
        out_ = numpy.all(numpy.asarray(out_.coefficients), out=out[0], axis=0)
    return out_
