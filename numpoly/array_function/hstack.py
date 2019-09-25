"""Stack arrays in sequence horizontally (column wise)."""
import numpy
import numpoly

from .common import implements


@implements(numpy.hstack)
def hstack(tup):
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Args:
        tup (Sequence[numpoly.ndpoly]):
            The arrays must have the same shape along all but the second axis,
            except 1-D arrays which can be any length.

    Returns:
        (numpoly.ndpoly):
            The array formed by stacking the given arrays.

    Examples:
        >>> a = numpoly.symbols("x y z")
        >>> b = numpoly.polynomial([1, 2, 3])
        >>> numpoly.hstack([a, b])
        polynomial([x, y, z, 1, 2, 3])
        >>> c = numpoly.polynomial([[1], [2], [3]])
        >>> d = a.reshape(3, 1)
        >>> numpoly.hstack([c, d])
        polynomial([[1, x],
                    [2, y],
                    [3, z]])

    """
    arrs = numpoly.atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return numpoly.concatenate(arrs, 0)
    return numpoly.concatenate(arrs, 1)
