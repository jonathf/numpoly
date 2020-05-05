"""Return elements chosen from `x` or `y` depending on `condition`."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.where)
def where(condition, *args):
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
        When only `condition` is provided, this function is a shorthand for
        ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
        preferred, as it behaves correctly for subclasses. The rest of this
        documentation covers only the case where all three arguments a re
        provided.

    Args:
        condition (numpy.ndarray, bool):
            Where True, yield `x`, otherwise yield `y`.
        x (numpoly.ndpoly): array_like
            Values from which to choose. `x`, `y` and `condition` need to be
            broadcastable to some shape.

    Returns:
        (numpoly.ndpoly):
            An array with elements from `x` where `condition` is True, and
            elements from `y` elsewhere.

    Examples:
        >>> poly = numpoly.symbols("x")*numpy.arange(4)
        >>> poly
        polynomial([0, x, 2*x, 3*x])
        >>> numpoly.where([1, 0, 1, 0], 7, 2*poly)
        polynomial([7, 2*x, 7, 6*x])
        >>> numpoly.where(poly, 2*poly, 4)
        polynomial([4, 2*x, 4*x, 6*x])
        >>> numpoly.where(poly)
        (array([1, 2, 3]),)

    """
    if isinstance(condition, numpoly.ndpoly):
        condition = numpy.any(condition.coefficients, 0).astype(bool)
    if not args:
        return numpy.where(condition)

    poly1, poly2 = numpoly.align_polynomials(*args)
    out = numpy.where(condition, poly1.values, poly2.values)
    out = numpoly.polynomial(out, names=poly1.indeterminants)
    return out
