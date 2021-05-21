"""Split array into multiple sub-arrays along the 3rd axis (depth)."""
from __future__ import annotations
from typing import List

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.dsplit)
def dsplit(
    ary: PolyLike,
    indices_or_sections: numpy.typing.ArrayLike,
) -> List[ndpoly]:
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    Examples:
        >>> poly = numpoly.monomial(8).reshape(2, 2, 2)
        >>> poly
        polynomial([[[1, q0],
                     [q0**2, q0**3]],
        <BLANKLINE>
                    [[q0**4, q0**5],
                     [q0**6, q0**7]]])
        >>> part1, part2 = numpoly.dsplit(poly, 2)
        >>> part1
        polynomial([[[1],
                     [q0**2]],
        <BLANKLINE>
                    [[q0**4],
                     [q0**6]]])
        >>> part2
        polynomial([[[q0],
                     [q0**3]],
        <BLANKLINE>
                    [[q0**5],
                     [q0**7]]])

    """
    ary = numpoly.aspolynomial(ary)
    results = numpy.dsplit(ary.values, indices_or_sections=indices_or_sections)
    return [
        numpoly.polynomial(
            result, names=ary.indeterminants, allocation=ary.allocation)
        for result in results
    ]
