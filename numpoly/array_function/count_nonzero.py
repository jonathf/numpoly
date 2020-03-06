"""Counts the number of non-zero values in the array a."""
import numpy
import numpoly

from .common import implements

@implements(numpy.count_nonzero)
def count_nonzero(x, axis=None, **kwargs):
    """
    Count the number of non-zero values in the array a.

    Args:
        x (numpoly.ndpoly):
            The array for which to count non-zeros.
        axis: (Union[int, Tuple[int], None]):
            Axis or tuple of axes along which to count non-zeros. Default is
            None, meaning that non-zeros will be counted along a flattened
            version of a.

    Returns:
        count (Union[bool, numpy.ndarray]):
            Number of non-zero values in the array along a given axis.
            Otherwise, the total number of non-zero values in the array is
            returned.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> numpoly.count_nonzero([x])
        1
        >>> numpoly.count_nonzero([[0,x,x*x,0,0],[x+1,0,0,2*x,19*x]])
        5
        >>> numpoly.count_nonzero([[0,x,7*x,0,0],[3*y,0,0,2,19*x+y]], axis=0)
        array([1, 1, 1, 1, 1])
        >>> numpoly.count_nonzero([[0,x,y,0,0],[x,0,0,2*x,19*y]], axis=1)
        array([2, 3])

    """
    a = numpoly.aspolynomial(x)

    return numpy.count_nonzero(numpy.any(a.coefficients, axis=0), axis=axis)
