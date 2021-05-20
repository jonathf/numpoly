"""View inputs as arrays with at least two dimensions."""
from __future__ import annotations
from typing import List, Union

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.atleast_2d)
def atleast_2d(*arys: PolyLike) -> Union[ndpoly, List[ndpoly]]:
    """
    View inputs as arrays with at least two dimensions.

    Args:
        arys:
            One or more array-like sequences. Non-array inputs are converted
            to arrays. Arrays that already have two or more dimensions are
            preserved.

    Returns:
        An array, or list of arrays, each with ``a.ndim >= 2``. Copies are
        avoided where possible, and views with two or more dimensions are
        returned.

    Examples:
        >>> numpoly.atleast_2d(numpoly.variable())
        polynomial([[q0]])
        >>> numpoly.atleast_2d(1, [2], [[3]])
        [polynomial([[1]]), polynomial([[2]]), polynomial([[3]])]

    """
    if len(arys) == 1:
        poly = numpoly.aspolynomial(arys[0])
        array = numpy.atleast_2d(poly.values)
        return numpoly.aspolynomial(array, names=poly.indeterminants)
    return [atleast_2d(ary) for ary in arys]
