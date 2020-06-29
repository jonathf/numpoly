"""Split an array into multiple sub-arrays."""
import numpy
import numpoly

from ..dispatch import implements


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
        polynomial([[1, q0, q0**2, q0**3],
                    [q0**4, q0**5, q0**6, q0**7]])
        >>> parts = numpoly.array_split(poly, 3, axis=1)
        >>> part1, part2, part3 = parts
        >>> part1
        polynomial([[1, q0],
                    [q0**4, q0**5]])
        >>> part2
        polynomial([[q0**2],
                    [q0**6]])
        >>> part3
        polynomial([[q0**3],
                    [q0**7]])

    """
    ary = numpoly.aspolynomial(ary)
    results = numpy.array_split(
        ary.values, indices_or_sections=indices_or_sections, axis=axis)
    return [numpoly.aspolynomial(result, names=ary.indeterminants)
            for result in results]
