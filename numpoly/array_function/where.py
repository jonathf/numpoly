"""Return elements chosen from `x` or `y` depending on `condition`."""

from __future__ import annotations

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.where)
def where(condition: numpy.typing.ArrayLike, *args: PolyLike) -> ndpoly:
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
        When only `condition` is provided, this function is a shorthand for
        ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
        preferred, as it behaves correctly for subclasses. The rest of this
        documentation covers only the case where all three arguments a re
        provided.

    Args:
        condition:
            Where True, yield `x`, otherwise yield `y`.
        x:
            Values from which to choose. `x`, `y` and `condition` need to be
            broadcastable to some shape.

    Return:
        An array with elements from `x` where `condition` is True,
        and elements from `y` elsewhere.

    Example:
        >>> poly = numpoly.variable()*numpy.arange(4)
        >>> poly
        polynomial([0, q0, 2*q0, 3*q0])
        >>> numpoly.where([1, 0, 1, 0], 7, 2*poly)
        polynomial([7, 2*q0, 7, 6*q0])
        >>> numpoly.where(poly, 2*poly, 4)
        polynomial([4, 2*q0, 4*q0, 6*q0])
        >>> numpoly.where(poly)
        (array([1, 2, 3]),)

    """
    if isinstance(condition, numpoly.ndpoly):
        condition = numpy.any(numpy.asarray(condition.coefficients), 0).astype(bool)
    if not args:
        return numpy.where(condition)

    poly1, poly2 = numpoly.align_polynomials(*args)
    coefficients = [
        numpy.where(condition, x1, x2)
        for x1, x2 in zip(poly1.coefficients, poly2.coefficients)
    ]
    dtype = numpy.result_type(poly1.dtype, poly2.dtype)
    return numpoly.polynomial_from_attributes(
        exponents=poly1.exponents,
        coefficients=coefficients,
        names=poly1.names,
        dtype=dtype,
    )
