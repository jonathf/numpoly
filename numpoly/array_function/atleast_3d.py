"""View inputs as arrays with at least three dimensions."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.atleast_3d)
def atleast_3d(*arys):
    """
    View inputs as arrays with at least three dimensions.

    Args:
        arys (numpoly.ndpoly):
            One or more array-like sequences. Non-array inputs are converted
            to arrays. Arrays that already have three or more dimensions are
            preserved.

    Returns:
        (numpoly.ndpoly):
            An array, or list of arrays, each with ``a.ndim >= 3``.  Copies are
            avoided where possible, and views with three or more dimensions are
            returned.  For example, a 1-D array of shape ``(N,)`` becomes
            a view of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)``
            becomes a view of shape ``(M, N, 1)``.

    Examples:
        >>> numpoly.atleast_3d(numpoly.variable())
        polynomial([[[q0]]])
        >>> a, b = numpoly.atleast_3d(1, [2, 3])
        >>> a
        polynomial([[[1]]])
        >>> b
        polynomial([[[2],
                     [3]]])

    """
    if len(arys) == 1:
        poly = numpoly.aspolynomial(arys[0])
        array = numpy.atleast_3d(poly.values)
        return numpoly.aspolynomial(array, names=poly.indeterminants)
    return [atleast_3d(ary) for ary in arys]
