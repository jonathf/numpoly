"""Return (x1 == x2) element-wise."""
import numpy
import numpoly

from .common import implements


@implements(numpy.equal)
def equal(x1, x2, out=None, where=True, **kwargs):
    """
    Return (x1 == x2) element-wise.

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
            Elsewhere, the `out` array will retain its original value.
            Note that if an uninitialized `out` array is created via the
            default ``out=None``, locations within it where the condition is
            False will remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        (Union[numpy.ndarray, numpy.generic]):
            Output array, element-wise comparison of `x1` and `x2`. Typically
            of type bool, unless ``dtype=object`` is passed. This is a scalar
            if both `x1` and `x2` are scalars.

    Examples:
        >>> x, y, z = xyz = numpoly.symbols("x y z")
        >>> numpoly.equal(xyz, x)
        array([ True, False, False])
        >>> numpoly.equal(xyz, [y, y, z])
        array([False,  True,  True])
        >>> numpoly.equal(x, y)
        array(False)

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    if out is None:
        out = numpy.ones(x1.shape, dtype=bool)
    for coeff1, coeff2 in zip(x1.coefficients, x2.coefficients):
        out &= numpy.equal(coeff1, coeff2, where=where, **kwargs)
    return out
