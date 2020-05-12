"""Return the truth value of (x1 >= x2) element-wise."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.greater_equal)
def greater_equal(x1, x2, out=None, **kwargs):
    """
    Return the truth value of (x1 >= x2) element-wise.

    Args:
        x1, x2 (numpoly.ndpoly):
            Input arrays. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the
            output).
        out (Optional[numpy.ndarray]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (Optional[numpy.ndarray]):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        out (numpy.ndarray, scalar):
            Output array, element-wise comparison of `x1` and `x2`. Typically
            of type bool, unless ``dtype=object`` is passed. This is a scalar
            if both `x1` and `x2` are scalars.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> numpoly.greater_equal(3, 4)
        array(False)
        >>> numpoly.greater_equal(4*x, 3*x)
        array(True)
        >>> numpoly.greater_equal(x, y)
        array(False)
        >>> numpoly.greater_equal(x**2, x)
        array(True)
        >>> numpoly.greater_equal([1, x, x**2, x**3], y)
        array([False, False,  True,  True])
        >>> numpoly.greater_equal(x+1, x-1)
        array(True)
        >>> numpoly.greater_equal(x, x)
        array(True)

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    coefficients1 = x1.coefficients
    coefficients2 = x2.coefficients
    if out is None:
        out = numpy.greater_equal(coefficients1[0], coefficients2[0], **kwargs)
    if not out.shape:
        return numpy.array(greater_equal(x1.ravel(), x2.ravel(), out=out.ravel()).item())
    for idx in numpoly.bsort(x1.exponents.T, ordering="GR"):
        indices = (coefficients1[idx] != 0) | (coefficients2[idx] != 0)
        indices &= coefficients1[idx] != coefficients2[idx]
        out[indices] = numpy.greater_equal(
            coefficients1[idx][indices], coefficients2[idx][indices], **kwargs)
    return out