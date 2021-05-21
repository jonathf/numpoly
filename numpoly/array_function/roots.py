"""Return the roots of a polynomial."""
from __future__ import annotations
import numpy
import numpoly

from ..baseclass import PolyLike
# from ..dispatch import implements


# @implements(numpy.roots)
def roots(poly: PolyLike) -> numpy.ndarray:
    """
    Return the roots of a polynomial.

    Assumes the polynomial has a single dimension.

    Args:
        poly:
            Polynomial to take roots of, or if constant, the coefficients of
            said polynomial. This to be compatible with :func:`numpy.roots`.

    Returns:
        An array containing the roots of the polynomial.

    Raises:
        ValueError:
            When `poly` cannot be converted to a rank-1 polynomial.

    Notes:
        The algorithm relies on computing the eigenvalues of the companion
        matrix [1]_.

    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.

    Examples:
        >>> q0 = numpoly.variable()
        >>> poly = 3.2*q0**2+2*q0+1
        >>> numpoly.roots(poly)
        array([-0.3125+0.46351241j, -0.3125-0.46351241j])
        >>> numpoly.roots([3.2, 2, 1])
        array([-0.3125+0.46351241j, -0.3125-0.46351241j])

    """
    # backwards compatibility
    poly = numpoly.aspolynomial(poly)
    if poly.isconstant():
        return numpy.roots(poly.tonumpy())
    # only rank-1
    if len(poly.names) > 1:
        raise ValueError("polynomial is not of rank 1.")
    # align exponents to include all coefficients
    filled_basis = poly.indeterminants**numpy.arange(
        numpoly.lead_exponent(poly), dtype=int)
    _, poly = numpoly.align_exponents(filled_basis, poly)
    # pass coefficients to numpy
    return numpy.roots(poly.coefficients[::-1])
