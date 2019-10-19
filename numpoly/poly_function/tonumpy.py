"""Cast polynomial to numpy.ndarray, if possible."""
import numpy
import numpoly


def tonumpy(poly):
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
        >>> numpoly.tonumpy(numpoly.polynomial([1, 2]))
        array([1, 2])
        >>> numpoly.tonumpy(numpoly.symbols("x"))
        Traceback (most recent call last):
            ...
        ValueError: only constant polynomials can be converted to array.

    """
    poly = numpoly.aspolynomial(poly)
    if not poly.isconstant():
        raise ValueError(
            "only constant polynomials can be converted to array.")
    idx = numpy.argwhere(numpy.all(poly.exponents == 0, -1)).item()
    return numpy.array(poly.coefficients[idx])
