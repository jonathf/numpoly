"""Join a sequence of arrays along a new axis."""
import numpy
import numpoly

from .common import implements


@implements(numpy.stack)
def stack(arrays, axis=0, out=None):
    """
    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    Args:
        arrays (Sequence[numpoly.ndpoly]):
            Each array must have the same shape.
        axis (Optional[int]):
            The axis in the result array along which the input arrays are
            stacked.
        out (Optional[numpy.ndarray]):
            If provided, the destination to place the result. The shape must be
            correct, matching that of what stack would have returned if no out
            argument were specified.

    Returns:
        (numpoly.ndpoly):
            The stacked array has one more dimension than the input arrays.

    Examples:
        >>> a = numpoly.symbols("x y z")
        >>> b = numpoly.polynomial([1, 2, 3])
        >>> numpoly.stack([a, b])
        polynomial([[x, y, z],
                    [1, 2, 3]])
        >>> numpoly.stack([a, b], axis=-1)
        polynomial([[x, 1],
                    [y, 2],
                    [z, 3]])

    """
    arrays = numpoly.align_exponents(*arrays)
    arrays = numpoly.align_dtype(*arrays)
    result = numpy.stack(
        [array.values for array in arrays], axis=axis, out=out)
    return numpoly.aspolynomial(result, names=arrays[0].indeterminants)

