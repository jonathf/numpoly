"""Check if a polynomial is constant or not."""

from __future__ import annotations

import numpy
import numpoly

from ..baseclass import PolyLike


def isconstant(poly: PolyLike) -> bool:
    """
    Check if a polynomial is constant or not.

    Args:
        poly:
            polynomial to check if is constant or not.

    Return:
        True if polynomial has no indeterminants.

    Example:
        >>> q0 = numpoly.variable()
        >>> numpoly.isconstant(numpoly.polynomial([q0]))
        False
        >>> numpoly.isconstant(numpoly.polynomial([1]))
        True

    """
    poly = numpoly.aspolynomial(poly)
    for exponent, coefficient in zip(poly.exponents, poly.coefficients):
        if not numpy.any(exponent):
            continue
        if numpy.any(coefficient):
            return False
    return True
