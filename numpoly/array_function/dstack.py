"""Stack arrays in sequence depth wise (along third axis)."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.dstack)
def dstack(tup):
    """
    Stack arrays in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Args:
        tup (Sequence[numpoly.ndpoly]):
            The arrays must have the same shape along all but the third axis.
            1-D or 2-D arrays must have the same shape.

    Returns:
        (numpoly.ndpoly):
            The array formed by stacking the given arrays, will be at least
            3-D.

    Examples:
        >>> poly1 = numpoly.variable(3)
        >>> const1 = numpoly.polynomial([1, 2, 3])
        >>> numpoly.dstack([poly1, const1])
        polynomial([[[q0, 1],
                     [q1, 2],
                     [q2, 3]]])
        >>> const2 = numpoly.polynomial([[1], [2], [3]])
        >>> poly2 = poly1.reshape(3, 1)
        >>> numpoly.dstack([const2, poly2])
        polynomial([[[1, q0]],
        <BLANKLINE>
                    [[2, q1]],
        <BLANKLINE>
                    [[3, q2]]])

    """
    arrays = numpoly.align_exponents(*tup)
    arrays = numpoly.align_dtype(*arrays)
    result = numpy.dstack([array.values for array in arrays])
    return numpoly.aspolynomial(result, names=arrays[0].indeterminants)
