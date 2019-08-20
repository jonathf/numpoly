import numpy

from .align import align_polynomials
from . import baseclass


def polynomial(poly_like, dtype=None):
    """
    Polynomial representation in variable dimensions.

    Examples:
        >>> print(polynomial({(1,): 1}))
        q0
        >>> x, y = numpoly.variable(2)
        >>> print(x**2 + x*y + 2)
        2+q0q1+q0^2
        >>> poly = -3*x + x**2 + y
        >>> print(polynomial([x*y, x, y]))
        [q0q1 q0 q1]
        >>> print(polynomial([1, 2, 3]))
        [1 2 3]
    """
    if poly_like is None:
        poly = baseclass.ndpoly(keys=[(0,)], shape=())
        poly["0"] = 0

    elif isinstance(poly_like, dict):
        poly = baseclass.ndpoly(keys=[(0,)], shape=())
        exponents, coefficients = zip(*list(poly_like.items()))
        poly = polynomial_from_attributes(exponents, coefficients, dtype)

    elif isinstance(poly_like, baseclass.ndpoly):
        poly = poly_like.copy()

    elif isinstance(poly_like, (int, float, numpy.ndarray, numpy.generic)):
        poly = baseclass.ndpoly(keys=[(0,)], shape=())
        poly = polynomial_from_attributes([(0,)], numpy.array([poly_like]))

    else:
        poly = baseclass.ndpoly(keys=[(0,)], shape=())
        poly = compose_polynomial_array(poly_like)

    return poly


def polynomial_from_attributes(exponents, coefficients, dtype=None):
    exponents, coefficients = clean_attributes(exponents, coefficients)
    dtype = coefficients[0].dtype if dtype is None else dtype
    poly = baseclass.ndpoly(
        keys=exponents, shape=coefficients[0].shape, dtype=dtype)
    for key, values in zip(poly._keys, coefficients):
        poly[key] = values
    return poly


def clean_attributes(exponents, coefficients):
    coefficients = [numpy.asarray(coefficient)
                    for coefficient in coefficients]
    exponents, coefficients = zip(*[
        (exponent, coefficient)
        for exponent, coefficient in zip(exponents, coefficients)
        if numpy.any(coefficient) or not any(exponent)
    ])
    exponents = numpy.asarray(exponents, dtype=int)
    assert len(exponents.shape) == 2, exponents
    assert len(exponents) == len(coefficients)

    assert len(numpy.unique(exponents, axis=0)) == exponents.shape[0]

    return exponents, coefficients


def compose_polynomial_array(arrays):
    arrays = numpy.array(arrays, dtype=object)
    shape = arrays.shape
    arrays = arrays.flatten().tolist()

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

    return polynomial_from_attributes(exponents, coefficients)


def zeros(keys, shape, dtype=None):
    poly = baseclass.ndpoly(keys=keys, shape=shape, dtype=dtype)
    for key in poly._keys:
        poly[key] = 0
    return poly
