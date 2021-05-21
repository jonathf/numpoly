"""Evenly round to the given number of decimals."""
from __future__ import annotations
from typing import Optional

import numpy

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements, simple_dispatch


@implements(numpy.around, numpy.round)
def around(
    a: PolyLike,
    decimals: int = 0,
    out: Optional[ndpoly] = None,
) -> ndpoly:
    """
    Evenly round to the given number of decimals.

    Args:
        a:
            Input data.
        decimals:
            Number of decimal places to round to (default: 0). If decimals is
            negative, it specifies the number of positions to the left of the
            decimal point.
        out:
            Alternative output array in which to place the result. It must have
            the same shape as the expected output, but the type of the output
            values will be cast if necessary.

    Returns:
        An array of the same type as `a`, containing the rounded values. Unless
        `out` was specified, a new array is created.  A reference to the result
        is returned. The real and imaginary parts of complex numbers are
        rounded separately.  The result of rounding a float is a float.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.around([0.37, 1.64*q0-2.45])
        polynomial([0.0, 2.0*q0-2.0])
        >>> numpoly.around([0.37, 1.64*q0-23.45], decimals=1)
        polynomial([0.4, 1.6*q0-23.4])
        >>> numpoly.around([0.37, 1.64*q0-23.45], decimals=-1)
        polynomial([0.0, -20.0])

    """
    return simple_dispatch(
        numpy_func=numpy.around,
        inputs=(a,),
        out=None if out is None else (out,),
        decimals=decimals,
    )
