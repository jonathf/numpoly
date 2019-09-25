"""Convert inputs to arrays with at least one dimension."""
import numpy
import numpoly

from .common import implements


@implements(numpy.atleast_1d)
def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Args:
        arys (numpoly.ndpoly):
            One or more input arrays.

    Returns:
        (ndarray):
            An array, or list of arrays, each with ``a.ndim >= 1``. Copies are
            made only if necessary.

    Examples:
        >>> numpoly.atleast_1d(numpoly.symbols("x"))
        polynomial([x])
        >>> numpoly.atleast_1d(1, [2, 3])
        [polynomial([1]), polynomial([2, 3])]

    """
    arys = [numpoly.aspolynomial(ary) for ary in arys]
    arys = [ary if ary.ndim else ary.reshape(1) for ary in arys]
    if len(arys) == 1:
        return arys[0]
    return arys
