"""Split an array into multiple sub-arrays vertically (row-wise)."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.vsplit)
def vsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays vertically (row-wise).

    Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
    to ``split`` with `axis=0` (default), the array is always split along the
    first axis regardless of the array dimension.

    See Also:
        split : Split an array into multiple sub-arrays of equal size.

    Examples:
        >>> poly = numpoly.monomial(8).reshape(4, 2)
        >>> poly
        polynomial([[1, q0],
                    [q0**2, q0**3],
                    [q0**4, q0**5],
                    [q0**6, q0**7]])
        >>> part1, part2 = numpoly.vsplit(poly, 2)
        >>> part1
        polynomial([[1, q0],
                    [q0**2, q0**3]])
        >>> part2
        polynomial([[q0**4, q0**5],
                    [q0**6, q0**7]])
        >>> part1, part2, part3 = numpoly.vsplit(poly, [1, 2])
        >>> part1
        polynomial([[1, q0]])
        >>> part3
        polynomial([[q0**4, q0**5],
                    [q0**6, q0**7]])

    """
    ary = numpoly.aspolynomial(ary)
    results = numpy.vsplit(ary.values, indices_or_sections=indices_or_sections)
    return [
        numpoly.polynomial(
            result, names=ary.indeterminants, allocation=ary.allocation)
        for result in results
    ]
