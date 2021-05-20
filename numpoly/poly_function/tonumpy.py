"""Cast polynomial to numpy.ndarray, if possible."""
from __future__ import annotations

import numpy
import numpoly

from ..baseclass import PolyLike


def tonumpy(poly: PolyLike) -> numpy.ndarray:
    """
    Cast polynomial to numpy.ndarray, if possible.

    Args:
        poly:
            polynomial to cast.

    Returns:
        Numpy array.

    Raises:
        numpoly.baseclass.FeatureNotSupported:
            Only constant polynomials can be cast to numpy.ndarray.

    Examples:
        >>> numpoly.tonumpy(numpoly.polynomial([1, 2]))
        array([1, 2])

    """
    poly = numpoly.aspolynomial(poly)
    if not poly.isconstant():
        raise numpoly.FeatureNotSupported(
            "only constant polynomials can be converted to array.")
    idx = numpy.argwhere(numpy.all(poly.exponents == 0, -1)).item()
    if poly.size:
        return numpy.array(poly.coefficients[idx])
    return numpy.array([])
