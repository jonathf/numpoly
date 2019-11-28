"""Return a true division of the inputs, element-wise."""
from __future__ import division

import numpy
import numpoly

from .common import implements


@implements(numpy.divide, numpy.true_divide)
def divide(x1, x2, out=None, where=True, **kwargs):
    """
    Return a true division of the inputs, element-wise.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Args:
        x1 (numpoly.ndpoly):
            Dividend array.
        x2 (numpoly.ndpoly):
            Divisor array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a commo n shape (which becomes the shape of the
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
        (numpoly.ndpoly):
            This is a scalar if both `x1` and `x2` are scalars.

    Examples:
        >>> xyz = numpoly.symbols("x y z")
        >>> numpoly.divide(xyz, 4)
        polynomial([0.25*x, 0.25*y, 0.25*z])
        >>> numpoly.divide(xyz, [1, 2, 4])
        polynomial([x, 0.5*y, 0.25*z])
        >>> numpoly.divide([1, 2, 4], xyz)
        Traceback (most recent call last):
            ...
        ValueError: only constant polynomials can be converted to array.

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    x2 = x2.tonumpy()
    no_output = out is None
    if no_output:
        out = numpoly.ndpoly(
            exponents=x1.exponents,
            shape=x1.shape,
            names=x1.indeterminants,
            dtype=numpy.common_type(x1, numpy.array(1.)),
        )
    elif not isinstance(out, numpy.ndarray):
        assert len(out) == 1, "only one output"
        out = out[0]
    for key in x1.keys:
        out[key] = 0
        numpy.true_divide(x1[key], x2, out=out[key], where=where, **kwargs)
    if no_output:
        out = numpoly.clean_attributes(out)
    return out
