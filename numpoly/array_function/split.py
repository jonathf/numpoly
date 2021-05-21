"""Split an array into multiple sub-arrays."""
from __future__ import annotations
from typing import List

import numpy
import numpy.typing
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.split)
def split(
    ary: PolyLike,
    indices_or_sections: numpy.typing.ArrayLike,
    axis: int = 0,
) -> List[ndpoly]:
    """
    Split an array into multiple sub-arrays.

    Args:
        ary:
            Array to be divided into sub-arrays.
        indices_or_sections:
            If `indices_or_sections` is an integer, N, the array will be
            divided into N equal arrays along `axis`.  If such a split is not
            possible, an error is raised.

            If `indices_or_sections` is a 1-D array of sorted integers, the
            entries indicate where along `axis` the array is split.  For
            example, ``[2, 3]`` would, for ``axis=0``, result in

            - ary[:2]
            - ary[2:3]
            - ary[3:]

            If an index exceeds the dimension of the array along `axis`, an
            empty sub-array is returned correspondingly.
        axis:
            The axis along which to split, default is 0.

    Returns:
        A list of sub-arrays.

    Raises:
        ValueError:
            If `indices_or_sections` is given as an integer, but a split does
            not result in equal division.

    See Also:
        array_split:
            Split an array into multiple sub-arrays of equal or near-equal
            size. Does not raise an exception if an equal division cannot be
            made.
        hsplit:
            Split array into multiple sub-arrays horizontally (column-wise).
        vsplit:
            Split array into multiple sub-arrays vertically (row wise).
        dsplit:
            Split array into multiple sub-arrays along the 3rd axis (depth).

    Examples:
        >>> poly = numpoly.monomial(16).reshape(4, 4)
        >>> poly
        polynomial([[1, q0, q0**2, q0**3],
                    [q0**4, q0**5, q0**6, q0**7],
                    [q0**8, q0**9, q0**10, q0**11],
                    [q0**12, q0**13, q0**14, q0**15]])
        >>> part1, _ = numpoly.split(poly, 2, axis=0)
        >>> part1
        polynomial([[1, q0, q0**2, q0**3],
                    [q0**4, q0**5, q0**6, q0**7]])
        >>> part1, _ = numpoly.split(poly, 2, axis=1)
        >>> part1
        polynomial([[1, q0],
                    [q0**4, q0**5],
                    [q0**8, q0**9],
                    [q0**12, q0**13]])

    """
    ary = numpoly.aspolynomial(ary)
    results = numpy.split(
        ary.values, indices_or_sections=indices_or_sections, axis=axis)
    return [numpoly.aspolynomial(result, names=ary.indeterminants)
            for result in results]
