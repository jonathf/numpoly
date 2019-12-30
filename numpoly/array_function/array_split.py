"""Split an array into multiple sub-arrays."""
import numpy
import numpoly

from .common import implements


@implements(numpy.array_split)
def array_split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays.

    Please refer to the ``split`` documentation.  The only difference between
    these functions is that ``array_split`` allows `indices_or_sections` to be
    an integer that does *not* equally divide the axis. For an array of length
    l that should be split into n sections, it returns l % n sub-arrays of size
    l//n + 1 and the rest of size l//n.

    See Also:
        split : Split an array into multiple sub-arrays of equal size.

    Examples:
        >>> poly = numpoly.monomial(8).reshape(2, 4)
        >>> poly
        polynomial([[1, q, q**2, q**3],
                    [q**4, q**5, q**6, q**7]])
        >>> part1, part2, part3 = numpoly.array_split(poly, 3, axis=1)
        >>> part1
        polynomial([[1, q],
                    [q**4, q**5]])
        >>> part2
        polynomial([[q**2],
                    [q**6]])
        >>> part3
        polynomial([[q**3],
                    [q**7]])

    """
    ary = numpoly.aspolynomial(ary)
    results = numpy.array_split(
        ary.values, indices_or_sections=indices_or_sections, axis=axis)
    return [numpoly.aspolynomial(result, names=ary.indeterminants)
            for result in results]
