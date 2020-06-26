"""Check if a polynomial is constant or not."""
import numpy
import numpoly


def isconstant(poly):
    """
    Check if a polynomial is constant or not.

    Args:
        poly (numpoly.ndpoly):
            polynomial to check if is constant or not.

    Returns:
        (bool):
            True if polynomial has no indeterminants.

    Examples:
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
