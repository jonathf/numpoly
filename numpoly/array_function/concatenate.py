"""Join a sequence of arrays along an existing axis."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.concatenate)
def concatenate(arrays, axis=0, out=None):
    """
    Join a sequence of arrays along an existing axis.

    Args:
        arrays (Iterable[numpoly.ndpoly]):
            The arrays must have the same shape, except in the dimension
            corresponding to `axis` (the first, by default).
        axis (Optional[int]):
            The axis along which the arrays will be joined.  If axis is None,
            arrays are flattened before use.  Default is 0.
        out (Optional[numpy.ndarray]):
            If provided, the destination to place the result. The shape must be
            correct, matching that of what concatenate would have returned if
            no out argument were specified.

    Returns:
        (numpoly.ndpoly):
            The concatenated array.

    Examples:
        >>> const = numpy.array([[1, 2], [3, 4]])
        >>> poly = numpoly.variable(2).reshape(1, 2)
        >>> numpoly.concatenate((const, poly), axis=0)
        polynomial([[1, 2],
                    [3, 4],
                    [q0, q1]])
        >>> numpoly.concatenate((const, poly.T), axis=1)
        polynomial([[1, 2, q0],
                    [3, 4, q1]])
        >>> numpoly.concatenate((const, poly), axis=None)
        polynomial([1, 2, 3, 4, q0, q1])

    """
    arrays = numpoly.align_exponents(*arrays)
    arrays = numpoly.align_dtype(*arrays)
    result = numpy.concatenate(
        [array.values for array in arrays], axis=axis, out=out)
    return numpoly.aspolynomial(result, names=arrays[0].indeterminants)
