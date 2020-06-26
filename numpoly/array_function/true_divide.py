"""Return true division of the inputs, element-wise."""
from __future__ import division

import numpy
import numpoly

from ..dispatch import implements_ufunc

DIVIDE_ERROR_MSG = """
Divisor in division is a polynomial.
Polynomial division differs from numerical division;
Use ``numpoly.poly_divide`` to get polynomial division."""


@implements_ufunc(numpy.true_divide)
def true_divide(x1, x2, out=None, where=True, **kwargs):
    """
    Return true division of the inputs, element-wise.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Args:
        x1 (numpoly.ndpoly):
            Dividend array.
        x2 (numpoly.ndpoly):
            Divisor array. If ``x1.shape != x2.shape``, they must be
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
        (numpoly.ndpoly):
            This is a scalar if both `x1` and `x2` are scalars.

    Raises:
        numpoly.baseclass.FeatureNotSupported:
            If `x2` contains indeterminants, numerical division is no longer
            possible and an error is raised instead. For polynomial
            division see ``numpoly.poly_divide``.

    Examples:
        >>> q0q1q2 = numpoly.variable(3)
        >>> numpoly.true_divide(q0q1q2, 4)
        polynomial([0.25*q0, 0.25*q1, 0.25*q2])
        >>> numpoly.true_divide(q0q1q2, [1, 2, 4])
        polynomial([q0, 0.5*q1, 0.25*q2])

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    if not x2.isconstant():
        raise numpoly.FeatureNotSupported(DIVIDE_ERROR_MSG)
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
