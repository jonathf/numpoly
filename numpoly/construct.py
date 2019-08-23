import numpy

from . import align, baseclass


def polynomial(
        poly_like,
        indeterminants="q",
        dtype=None,
):
    """
    Polynomial representation in variable dimensions.

    Examples:
        >>> print(polynomial({(1,): 1}))
        q
        >>> x, y = numpoly.symbols("x y")
        >>> print(x**2 + x*y + 2)
        2+x*y+x**2
        >>> poly = -3*x + x**2 + y
        >>> print(polynomial([x*y, x, y]))
        [x*y x y]
        >>> print(polynomial([1, 2, 3]))
        [1 2 3]
    """
    if poly_like is None:
        poly = baseclass.ndpoly(
            exponents=[(0,)],
            shape=(),
            indeterminants=indeterminants,
            dtype=dtype,
        )
        poly["0"] = 0

    elif isinstance(poly_like, dict):
        poly = baseclass.ndpoly(exponents=[(0,)], shape=())
        exponents, coefficients = zip(*list(poly_like.items()))
        poly = polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
            dtype=dtype,
        )

    elif isinstance(poly_like, baseclass.ndpoly):
        poly = poly_like.copy()

    elif isinstance(poly_like, (int, float, numpy.ndarray, numpy.generic)):
        poly = polynomial_from_attributes(
            exponents=[(0,)],
            coefficients=numpy.array([poly_like]),
            indeterminants=indeterminants,
            dtype=dtype,
        )

    else:
        poly = compose_polynomial_array(
            arrays=poly_like,
            dtype=dtype,
        )

    return poly


def polynomial_from_attributes(
        exponents,
        coefficients,
        indeterminants,
        dtype=None,
        trim=True,
):
    if trim:
        exponents, coefficients, indeterminants = clean_attributes(
            exponents, coefficients, indeterminants)
    dtype = coefficients[0].dtype if dtype is None else dtype
    poly = baseclass.ndpoly(
        exponents=exponents,
        shape=coefficients[0].shape,
        indeterminants=indeterminants,
        dtype=dtype,
    )
    for exponent, values in zip(poly._exponents, coefficients):
        poly[exponent] = values
    return poly


def clean_attributes(exponents, coefficients, indeterminants):
    coefficients = [numpy.asarray(coefficient)
                    for coefficient in coefficients]
    exponents, coefficients = zip(*[
        (exponent, coefficient)
        for exponent, coefficient in zip(exponents, coefficients)
        if numpy.any(coefficient) or not any(exponent)
    ])
    exponents = numpy.asarray(exponents, dtype=int)

    if isinstance(indeterminants, baseclass.ndpoly):
        indeterminants = indeterminants._indeterminants
    if isinstance(indeterminants, str):
        if exponents.shape[1] > 1:
            indeterminants = ["%s%d" % (indeterminants, idx)
                              for idx in range(exponents.shape[1])]
        else:
            indeterminants = [indeterminants]
    assert len(indeterminants) == exponents.shape[1], (indeterminants, exponents)

    indices = numpy.any(exponents != 0, 0)
    assert exponents.shape[1] == len(indices), (exponents, indices)
    if not numpy.any(indices):
        indices[0] = True
    exponents = exponents[:, indices]
    assert exponents.size, (exponents, indices)
    indeterminants = numpy.array(indeterminants)[indices].tolist()

    assert len(exponents.shape) == 2, exponents
    assert len(exponents) == len(coefficients)
    assert len(numpy.unique(exponents, axis=0)) == exponents.shape[0], exponents
    assert sorted(set(indeterminants)) == sorted(indeterminants)

    return exponents, coefficients, indeterminants


def compose_polynomial_array(
        arrays,
        dtype=None,
):
    arrays = numpy.array(arrays, dtype=object)
    shape = arrays.shape
    arrays = arrays.flatten()

    indices = numpy.array([isinstance(array, baseclass.ndpoly)
                           for array in arrays])
    arrays[indices] = align.align_polynomial_indeterminants(*arrays[indices])
    indeterminants = arrays[indices][0] if numpy.any(indices) else "q"
    arrays = arrays.tolist()

    dtypes = []
    keys = {(0,)}
    for array in arrays:
        if isinstance(array, baseclass.ndpoly):
            dtypes.append(array._dtype)
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
        if isinstance(array, baseclass.ndpoly):
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

    exponents, coefficients, indeterminants = clean_attributes(
        exponents, coefficients, indeterminants)
    return polynomial_from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        indeterminants=indeterminants,
    )


def zeros(exponents, shape, dtype=None):
    poly = baseclass.ndpoly(exponents=exponents, shape=shape, dtype=dtype)
    for exponent in poly._exponents:
        poly[exponent] = 0
    return poly
