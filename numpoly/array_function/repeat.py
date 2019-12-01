"""Repeat elements of an array."""
import numpy
import numpoly

from .common import implements


@implements(numpy.repeat)
def repeat(a, repeats, axis=0):
    """
    Repeat elements of an array.

    Args:
        a (numpoly.ndpoly):
            Input array.
        repeats (Union[int, numpy.ndarray]):
            The number of repetitions for each element. `repeats` is
            broadcasted to fit the shape of the given axis.
        axis (Optional[int]):
            The axis along which to repeat values. By default, use the
            flattened input array, and return a flat output array.

    Returns:
        (ndarray):
            Output array which has the same shape as `a`, except along the
            given axis.

    Examples:
        >>> x = numpoly.symbols("x")
        >>> numpoly.repeat(x, 4)
        polynomial([x, x, x, x])
        >>> poly = numpoly.polynomial([[1, x-1], [x**2, x]])
        >>> numpoly.repeat(poly, 2)
        polynomial([[1, -1+x],
                    [1, -1+x],
                    [x**2, x],
                    [x**2, x]])
        >>> numpoly.repeat(poly, 3, axis=1)
        polynomial([[1, 1, 1, -1+x, -1+x, -1+x],
                    [x**2, x**2, x**2, x, x, x]])
        >>> numpoly.repeat(poly, [1, 2], axis=0)
        polynomial([[1, -1+x],
                    [x**2, x],
                    [x**2, x]])

    """
    a = numpoly.aspolynomial(a)
    result = numpy.repeat(a.values, repeats=repeats, axis=axis)
    return numpoly.aspolynomial(result, names=a.indeterminants)
