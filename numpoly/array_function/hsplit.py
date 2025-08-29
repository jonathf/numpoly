"""Split an array into multiple sub-arrays horizontally (column-wise)."""

from __future__ import annotations
from typing import Sequence

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.hsplit)
def hsplit(
    ary: PolyLike,
    indices_or_sections: numpy.typing.ArrayLike,
) -> Sequence[ndpoly]:
    """
    Split an array into multiple sub-arrays horizontally (column-wise).

    Please refer to the `split` documentation.  `hsplit` is equivalent to
    `split` with ``axis=1``, the array is always split along the second axis
    regardless of the array dimension.

    Note:
        split : Split an array into multiple sub-arrays of equal size.

    Example:
        >>> poly = numpoly.monomial(8).reshape(2, 4)
        >>> poly
        polynomial([[1, q0, q0**2, q0**3],
                    [q0**4, q0**5, q0**6, q0**7]])
        >>> part1, part2 = numpoly.hsplit(poly, 2)
        >>> part1
        polynomial([[1, q0],
                    [q0**4, q0**5]])
        >>> part2
        polynomial([[q0**2, q0**3],
                    [q0**6, q0**7]])
        >>> part1, part2, part3 = numpoly.hsplit(poly, [1, 2])
        >>> part1
        polynomial([[1],
                    [q0**4]])
        >>> part3
        polynomial([[q0**2, q0**3],
                    [q0**6, q0**7]])

    """
    ary = numpoly.aspolynomial(ary)
    results = numpy.hsplit(ary.values, indices_or_sections=indices_or_sections)
    return [
        numpoly.polynomial(result, names=ary.indeterminants, allocation=ary.allocation)
        for result in results
    ]
