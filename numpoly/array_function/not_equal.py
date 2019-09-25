"""Return (x1 != x2) element-wise."""
import numpy
import numpoly

from .common import implements


@implements(numpy.not_equal)
def not_equal(x1, x2, out=None, where=True, **kwargs):
    """
    Return (x1 != x2) element-wise.

    Args:
        x1, x2 (numpoly.ndpoly):
            Input arrays.  If ``x1.shape != x2.shape``, they must be
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
        (numpy.ndarray):
            Output array, element-wise comparison of `x1` and `x2`. Typically
            of type bool, unless ``dtype=object`` is passed.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> numpoly.not_equal([x, x], [x, y])
        array([False,  True])
        >>> numpoly.not_equal([x, x], [[x, y], [y, x]])
        array([[False,  True],
               [ True, False]])

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    for key in x1.keys:
        tmp = numpy.not_equal(x1[key], x2[key], where=where, **kwargs)
        if out is None:
            out = tmp
        else:
            out |= tmp
    return out
