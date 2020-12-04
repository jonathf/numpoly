"""Compose polynomial from array of arrays of polynomials."""
import numpy
import numpoly


def compose_polynomial_array(
        arrays,
        dtype=None,
        allocation=None,
):
    """
    Compose polynomial from array of arrays of polynomials.

    Backend for `numpoly.polynomial` when input is undetermined.

    Args:
        arrays (Any):
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        dtype (Optional[numpy.dtype]):
            Data type used for the polynomial coefficients.
        allocation (Optional[int]):
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Return:
        (numpoly.ndpoly):
            Polynomial based on input `arrays`.
    """
    arrays_ = numpy.array(arrays, dtype=object)
    shape = arrays_.shape
    if not arrays_.size:
        return numpoly.ndpoly(shape=shape, dtype=dtype)
    if len(arrays_.shape) > 1:
        return numpoly.concatenate([
            numpoly.aspolynomial(array, dtype)[numpy.newaxis]
            for array in arrays
        ], axis=0)

    arrays = arrays_.flatten()

    indices = numpy.array([isinstance(array, numpoly.ndpoly)
                           for array in arrays])
    arrays[indices] = numpoly.align_indeterminants(*arrays[indices])
    names = arrays[indices][0] if numpy.any(indices) else None
    arrays = arrays.tolist()

    dtypes = []
    keys = {(0,)}
    for array in arrays:
        if isinstance(array, numpoly.ndpoly):
            dtypes.append(array.dtype)
            keys = keys.union([tuple(key) for key in array.exponents.tolist()])
        elif isinstance(array, (numpy.generic, numpy.ndarray)):
            dtypes.append(array.dtype)
        else:
            dtypes.append(type(array))

    if dtype is None:
        dtype = numpy.find_common_type(dtypes, [])
    length = max([len(key) for key in keys])

    collection = {}
    for idx, array in enumerate(arrays):
        if isinstance(array, numpoly.ndpoly):
            for key, value in zip(array.exponents, array.coefficients):
                key = tuple(key)+(0,)*(length-len(key))
                if key not in collection:
                    collection[key] = numpy.zeros(len(arrays), dtype=dtype)
                collection[key][idx] = value
        else:
            key = (0,)*length
            if key not in collection:
                collection[key] = numpy.zeros(len(arrays), dtype=dtype)
            collection[key][idx] = array

    exponents = sorted(collection)
    coefficients = numpy.array([collection[key] for key in exponents])
    coefficients = coefficients.reshape(-1, *shape)

    return numpoly.ndpoly.from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        names=names,
        allocation=allocation,
    )
