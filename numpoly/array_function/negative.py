"""Numerical negative, element-wise."""
import numpy
import numpoly

from ..dispatch import implements, simple_dispatch


@implements(numpy.negative)
def negative(x, out=None, where=True, **kwargs):
    """
    Numerical negative, element-wise.

    Args:
        x (numpoly.ndpoly):
            Input array.
        out (Optional[numpoly.ndpoly]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where : array_like, optional
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        y (numpoly.ndpoly):
            Returned array or scalar: `y = -x`. This is a scalar if `x` is
            a scalar.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.negative([-q0, q0])
        polynomial([q0, -q0])

    """
    return simple_dispatch(
        numpy_func=numpy.negative,
        inputs=(x,),
        out=out,
        where=where,
        **kwargs
    )
