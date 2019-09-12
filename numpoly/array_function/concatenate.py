"""Join a sequence of arrays along an existing axis."""
import numpy
import numpoly

from .common import implements


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
        >>> a = numpy.array([[1, 2], [3, 4]])
        >>> b = numpoly.symbols("x y").reshape(1, 2)
        >>> numpoly.concatenate((a, b), axis=0)
        polynomial([[1, 2],
                    [3, 4],
                    [x, y]])
        >>> numpoly.concatenate((a, b.T), axis=1)
        polynomial([[1, 2, x],
                    [3, 4, y]])
        >>> numpoly.concatenate((a, b), axis=None)
        polynomial([1, 2, 3, 4, x, y])

    """
    arrays = numpoly.align_indeterminants(*arrays)
    collections = [arg.todict() for arg in arrays]

    output = {}
    keys = {arg for collection in collections for arg in collection}
    for key in keys:
        values = [(collection[key] if key in collection
                   else numpy.zeros(array.shape, dtype=bool))
                  for collection, array in zip(collections, arrays)]
        output[key] = numpy.concatenate(values, axis=axis, out=out)

    exponents = sorted(output)
    coefficients = [output[exponent] for exponent in exponents]
    return numpoly.ndpoly.from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        indeterminants=arrays[0].indeterminants,
    )
