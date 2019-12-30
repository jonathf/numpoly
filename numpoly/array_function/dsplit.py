"""Split array into multiple sub-arrays along the 3rd axis (depth)."""
import numpy
import numpoly

from .common import implements


@implements(numpy.dsplit)
def dsplit(ary, indices_or_sections):
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    Examples:
        >>> poly = numpoly.monomial(8).reshape(2, 2, 2)
        >>> poly
        polynomial([[[1, q],
                     [q**2, q**3]],
        <BLANKLINE>
                    [[q**4, q**5],
                     [q**6, q**7]]])
        >>> part1, part2 = numpoly.dsplit(poly, 2)
        >>> part1
        polynomial([[[1],
                     [q**2]],
        <BLANKLINE>
                    [[q**4],
                     [q**6]]])
        >>> part2
        polynomial([[[q],
                     [q**3]],
        <BLANKLINE>
                    [[q**5],
                     [q**7]]])

    """
    ary = numpoly.aspolynomial(ary)
    results = numpy.dsplit(ary.values, indices_or_sections=indices_or_sections)
    return [numpoly.aspolynomial(result, names=ary.indeterminants)
            for result in results]
