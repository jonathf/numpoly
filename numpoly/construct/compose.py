"""Compose polynomial from array of arrays of polynomials."""
from __future__ import annotations
from typing import List, Optional, Sequence, Set, Tuple

import numpy
import numpy.typing

import numpoly
from ..baseclass import ndpoly, PolyLike


def compose_polynomial_array(
        arrays: Sequence[PolyLike],
        dtype: Optional[numpy.typing.DTypeLike] = None,
        allocation: Optional[int] = None,
) -> ndpoly:
    """
    Compose polynomial from array of arrays of polynomials.

    Backend for `numpoly.polynomial` when input is undetermined.

    Args:
        arrays:
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        dtype:
            Data type used for the polynomial coefficients.
        allocation:
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Return:
        Polynomial based on input `arrays`.

    """
    oarrays = numpy.array(arrays, dtype=object)
    shape = oarrays.shape
    if not oarrays.size:
        return numpoly.ndpoly(shape=shape, dtype=dtype)
    if len(oarrays.shape) > 1:
        return numpoly.concatenate([numpoly.expand_dims(
            numpoly.aspolynomial(array, dtype=dtype), axis=0)
                                    for array in arrays], axis=0)

    oarrays = oarrays.flatten()
    indices = numpy.array([isinstance(array, numpoly.ndpoly)
                           for array in oarrays])
    oarrays[indices] = numpoly.align_indeterminants(*oarrays[indices])
    names = oarrays[indices][0] if numpy.any(indices) else None
    oarrays = oarrays.tolist()

    dtypes: List[numpy.typing.DTypeLike] = []
    keys: Set[Tuple[int, ...]] = {(0,)}
    for array in oarrays:
        if isinstance(array, numpoly.ndpoly):
            dtypes.append(array.dtype)
            keys = keys.union({tuple(int(k) for k in key)
                               for key in array.exponents.tolist()})
        elif isinstance(array, (numpy.generic, numpy.ndarray)):
            dtypes.append(array.dtype)
        else:
            dtypes.append(type(array))

    if dtype is None:
        dtype = numpy.find_common_type(dtypes, [])
    length = max(1, max([len(key) for key in keys]))

    collection = {}
    for idx, array in enumerate(oarrays):
        if isinstance(array, numpoly.ndpoly):
            for key, value in zip(array.exponents, array.coefficients):
                key = tuple(key)+(0,)*(length-len(key))
                if key not in collection:
                    collection[key] = numpy.zeros(len(oarrays), dtype=dtype)
                collection[key][idx] = value
        else:
            key = (0,)*length
            if key not in collection:
                collection[key] = numpy.zeros(len(oarrays), dtype=dtype)
            collection[key][idx] = array

    exponents = sorted(collection)
    coefficients = numpy.array([collection[key] for key in exponents])
    coefficients = coefficients.reshape(-1, *shape)

    return numpoly.ndpoly.from_attributes(
        exponents=exponents,
        coefficients=list(coefficients),
        names=names,
        allocation=allocation,
    )
