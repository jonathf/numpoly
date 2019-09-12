"""Compute the outer product of two vectors."""
import numpy
import numpoly

from .common import implements


@implements(numpy.outer)
def outer(a, b, out=None):
    """
    Compute the outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``, the outer product is::

        [[a0*b0  a0*b1 ... a0*bN ]
         [a1*b0    .
         [ ...          .
         [aM*b0            aM*bN ]]

    Args:
        a (numpoly.ndpoly):
            First input vector. Input is flattened if not already
            1-dimensional.
        b (numpoly.ndpoly):
            Second input vector. Input is flattened if not already
            1-dimensional.
        out (numpy.ndarray):
            A location where the result is stored.

    Returns:
        (numpoly.ndpoly):
            ``out[i, j] = a[i] * b[j]``

    Examples:
        >>> numpoly.outer(numpoly.symbols("x y z"), numpy.arange(5))
        polynomial([[0, x, 2*x, 3*x, 4*x],
                    [0, y, 2*y, 3*y, 4*y],
                    [0, z, 2*z, 3*z, 4*z]])

    """
    a, b = numpoly.align_exponents(a, b)
    a = a.flatten()[:, numpy.newaxis]
    b = b.flatten()[numpy.newaxis, :]
    return numpoly.multiply(a, b, out=out)
