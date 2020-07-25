"""Return the element-wise square of the input."""
import numpy
import numpoly

from ..dispatch import implements
from .multiply import multiply


@implements(numpy.square)
def square(x, out=None, where=True, **kwargs):
    """
    Return the element-wise square of the input.

    Args:
        x (numpoly.ndpoly):
            Input data.
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
        out (numpoly.ndpoly):
            Element-wise `x*x`, of the same shape and dtype as `x`.
            This is a scalar if `x` is a scalar.

    Examples:
        >>> numpoly.square([1j, 1])
        polynomial([(-1+0j), (1+0j)])
        >>> poly = numpoly.sum(numpoly.variable(2))
        >>> poly
        polynomial(q1+q0)
        >>> numpoly.square(poly)
        polynomial(q1**2+2*q0*q1+q0**2)

    """
    return multiply(x, x, out=out, where=where, **kwargs)
