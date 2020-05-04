"""Compute the truth value of x1 OR x2 element-wise."""
import numpy
import numpoly

from ..dispatch import implements, simple_dispatch


@implements(numpy.logical_or)
def logical_or(x1, x2, out=None, where=True, **kwargs):
    """
    Compute the truth value of x1 OR x2 element-wise.

    Args:
        x1, x2 (numpoly.ndpoly):
            Logical OR is applied to the elements of `x1` and `x2`. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common
            shape (which becomes the shape of the output).
        out (Optional[numpy.ndarray]):
            A location into which the result is stored. If provided, it must have
            a shape that the inputs broadcast to. If not provided or `None`,
            a freshly-allocated array is returned. A tuple (possible only as a
            keyword argument) must have length equal to the number of outputs.
        where (Optional[numpy.ndarray]):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value.
            Note that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        y (numpy.ndarray):
            Boolean result of the logical OR operation applied to the elements
            of `x1` and `x2`; the shape is determined by broadcasting.
            This is a scalar if both `x1` and `x2` are scalars.

    Examples:
        >>> numpoly.logical_or(True, False)
        True
        >>> numpoly.logical_or([True, False], [False, False])
        array([ True, False])
        >>> x = numpy.arange(5)
        >>> numpoly.logical_or(x < 1, x > 3)
        array([ True, False, False, False,  True])

    """
    x1 = numpoly.aspolynomial(x1)
    x2 = numpoly.aspolynomial(x2)
    coefficients1 = numpy.any(x1.coefficients, 0)
    coefficients2 = numpy.any(x2.coefficients, 0)
    return numpy.logical_or(
        coefficients1, coefficients2, out=out, where=where, **kwargs)
