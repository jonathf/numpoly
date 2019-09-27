"""Cast polynomial to numpy.ndarray, if possible."""
import numpy
import numpoly


def toarray(poly):
    """
    Cast polynomial to numpy.ndarray, if possible.

    Args:
        poly (numpoly.ndpoly):
            polynomial to cast.

    Returns:
        (numpy.ndarray):
            Numpy array.

    Raises:
        ValueError:
            Only constant polynomials can be cast to numpy.ndarray.

    Examples:
        >>> numpoly.toarray(numpoly.polynomial([1, 2]))
        array([1, 2])
        >>> numpoly.toarray(numpoly.symbols("x"))
        Traceback (most recent call last):
            ...
        ValueError: only constant polynomials can be converted to array.

    """
    poly = numpoly.aspolynomial(poly)
    if not poly.isconstant():
        raise ValueError(
            "only constant polynomials can be converted to array.")
    return numpy.array(poly.coefficients[0])
