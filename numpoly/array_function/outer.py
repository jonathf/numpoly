"""Compute the outer product of two vectors."""
from __future__ import annotations
from typing import Optional

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.outer)
def outer(
    a: PolyLike,
    b: PolyLike,
    out: Optional[ndpoly] = None,
) -> ndpoly:
    """
    Compute the outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``, the outer product is::

        [[a0*b0  a0*b1 ... a0*bN ]
         [a1*b0    .
         [ ...          .
         [aM*b0            aM*bN ]]

    Args:
        a:
            First input vector. Input is flattened if not already
            1-dimensional.
        b:
            Second input vector. Input is flattened if not already
            1-dimensional.
        out:
            A location where the result is stored.

    Returns:
        ``out[i, j] = a[i] * b[j]``

    Examples:
        >>> poly = numpoly.variable(3)
        >>> const = numpy.arange(5)
        >>> numpoly.outer(poly, const)
        polynomial([[0, q0, 2*q0, 3*q0, 4*q0],
                    [0, q1, 2*q1, 3*q1, 4*q1],
                    [0, q2, 2*q2, 3*q2, 4*q2]])

    """
    a, b = numpoly.align_exponents(a, b)
    a = a.ravel()[:, numpy.newaxis]
    b = b.ravel()[numpy.newaxis, :]
    return numpoly.multiply(a, b, out=out)
