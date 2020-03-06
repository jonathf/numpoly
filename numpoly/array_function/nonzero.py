"""Return the indices of the elements that are non-zero."""
import numpy
import numpoly

from .common import implements

@implements(numpy.nonzero)
def nonzero(x, **kwargs):
    """
    Return the indices of the elements that are non-zero.

    Args:
        x (numpoly.ndpoly):
            Input array.

    Returns:
        indices (Union[bool, numpoly.ndarray]):
            Indices of elements that are non-zero.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> f = numpoly.polynomial([[3*x, 0, 0], [0, 4*y, 0], [5*x+y, 6*x, 0]])
        >>> f
        polynomial([[3*x, 0, 0],
                    [0, 4*y, 0],
                    [5*x+y, 6*x, 0]])
        >>> numpoly.nonzero(f)
        (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
        >>> f[numpoly.nonzero(f)]
        polynomial([3*x, 4*y, 5*x+y, 6*x])

    """
    a = numpoly.aspolynomial(x)

    return numpy.nonzero(numpy.any(a.coefficients, axis=0))
