"""Find the lead coefficients for each polynomial."""
from __future__ import annotations

import numpy
import numpoly

from ..baseclass import PolyLike


def lead_coefficient(
        poly: PolyLike,
        graded: bool = False,
        reverse: bool = False,
) -> numpy.ndarray:
    """
    Find the lead coefficients for each polynomial.

    As polynomials are not inherently sortable, values are sorted using the
    highest `lexicographical` ordering. Between the values that have the same
    highest ordering, the elements are sorted using the coefficients.

    Args:
        poly:
            Polynomial to locate coefficients on.
        graded:
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**3*q1*q2``,
            ``q0*q1**3*q2`` and ``q0*q1*z**3``.
        reverse:
            Reverses lexicographical sorting meaning that ``q0*q1**3`` is
            considered bigger than ``q0**3*q1``, instead of the opposite.

    Returns:
        Array of same shape and type as `poly`, containing all the lead
        coefficients.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.lead_coefficient(q0+2*q0**2+3*q0**3)
        3
        >>> numpoly.lead_coefficient([1-4*q1+q0, 2*q0**2-q1, 4])
        array([-4, -1,  4])

    """
    poly = numpoly.aspolynomial(poly)
    out = numpy.zeros(poly.shape, dtype=poly.dtype)
    if not out.size:
        return out
    for idx in numpoly.glexsort(
            poly.exponents.T, graded=graded, reverse=reverse):
        values = poly.coefficients[idx]
        indices = values != 0
        out[indices] = values[indices]
    if not poly.shape:
        out = out.item()
    return out
