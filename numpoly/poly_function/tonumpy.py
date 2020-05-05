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
