"""Convert inputs to arrays with at least one dimension."""
from __future__ import annotations
from typing import List, Union

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.atleast_1d)
def atleast_1d(*arys: PolyLike) -> Union[ndpoly, List[ndpoly]]:
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Args:
        arys:
            One or more input arrays.

    Returns:
        An array, or list of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    Examples:
        >>> numpoly.atleast_1d(numpoly.variable())
        polynomial([q0])
        >>> numpoly.atleast_1d(1, [2, 3])
        [polynomial([1]), polynomial([2, 3])]

    """
    if len(arys) == 1:
        poly = numpoly.aspolynomial(arys[0])
        array = numpy.atleast_1d(poly.values)
        return numpoly.aspolynomial(array, names=poly.indeterminants)
    return [atleast_1d(ary) for ary in arys]
