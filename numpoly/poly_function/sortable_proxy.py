"""Create a numerical proxy for a polynomial to allow compare."""
from __future__ import annotations

import numpy
import numpoly

from ..baseclass import PolyLike


def sortable_proxy(
    poly: PolyLike,
    graded: bool = False,
    reverse: bool = False,
) -> numpy.ndarray:
    """
    Create a numerical proxy for a polynomial to allow compare.

    As polynomials are not inherently sortable, values are sorted using the
    highest `lexicographical` ordering. Between the values that have the same
    highest ordering, the elements are sorted using the coefficients. This also
    ensures that the method behaves as expected with ``numpy.ndarray``.

    Args:
        poly:
            Polynomial to convert into something sortable.
        graded:
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**3*q1*q2``,
            ``q0*q1**3*q2`` and ``q0*q1*z**3``.
        reverse:
            Reverses lexicographical sorting meaning that ``q0*q1**3`` is
            considered bigger than ``q0**3*q1``, instead of the opposite.

    Returns:
        Integer array where ``a > b`` is retained for the giving rule of
        ``ordering``.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial(
        ...     [q0**2, 2*q0, 3*q1, 4*q0, 5])
        >>> numpoly.sortable_proxy(poly)
        array([3, 1, 4, 2, 0])
        >>> numpoly.sortable_proxy(poly, reverse=True)
        array([4, 2, 1, 3, 0])
        >>> numpoly.sortable_proxy([8, 4, 10, -100])
        array([2, 1, 3, 0])
        >>> numpoly.sortable_proxy([[8, 4], [10, -100]])
        array([[2, 1],
               [3, 0]])

    """
    poly = numpoly.aspolynomial(poly)
    coefficients = poly.coefficients
    proxy = numpy.tile(-1, poly.shape)
    largest = numpoly.lead_exponent(poly, graded=graded, reverse=reverse)

    for idx in numpoly.glexsort(
            poly.exponents.T, graded=graded, reverse=reverse):

        indices = numpy.all(largest == poly.exponents[idx], axis=-1)
        values = numpy.argsort(coefficients[idx][indices])
        proxy[indices] = numpy.argsort(values)+numpy.max(proxy)+1

    proxy = numpy.argsort(numpy.argsort(proxy.ravel())).reshape(proxy.shape)
    return proxy
