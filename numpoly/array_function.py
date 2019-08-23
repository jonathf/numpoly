from __future__ import division
import logging
import inspect
from functools import wraps

import numpy

from . import align, baseclass, construct, export

ARRAY_FUNCTIONS = {}


def implements(*numpy_functions):
    """Register an __array_function__ implementation for Polynomial objects."""

    defaults = None
    try:
        spec = inspect.signature(numpy_functions[0])
        defaults = {key: value.default
                    for key, value in spec.parameters.items()
                    if value != inspect._empty}

    except AttributeError:
        try:
            spec = inspect.getargspec(numpy_functions[0]).args
            defaults = dict(zip(spec.args, spec.defaults))
        except AttributeError as err:
            logger.exception("Python version not supported <=2.7,>=3.5")
            raise err

    except ValueError:
        pass

    def decorator(func):

        @wraps(func)
        def wrapper_function(*args, **kwargs):
            """Function to replace numpy library function wrappers."""
            if defaults is None:
                return func(*args, **kwargs)

            params = defaults.copy()
            for arg, name in zip(args, params):
                if name in kwargs:
                    raise TypeError(
                        "%s() got multiple values for argument '%s'" % (
                            func.__name__, name))
                params[name] = arg
            params.update(kwargs)
            return func(**params)

        for numpy_function in numpy_functions:
            ARRAY_FUNCTIONS[numpy_function] = wrapper_function

        return func

    return decorator


def process_parameters(poly, kwargs):
    coefficients = numpy.array(poly.coefficients)
    assert kwargs.get("out", None) is None, "object read-only"
    if "axis" in kwargs:
        axis = kwargs["axis"]
        if axis is None:
            coefficients = coefficients.reshape(len(coefficients), -1)
            axis = 1
        else:
            axis = axis+1 if axis >= 0 else len(coefficients.shape)+axis
        kwargs["axis"] = axis
    return coefficients, kwargs


@implements(numpy.abs, numpy.absolute)
def absolute(x, **kwargs):
    x = construct.polynomial(x)
    coefficients = numpy.absolute(x.coefficients, **kwargs)
    return construct.polynomial_from_attributes(
        x.exponents, coefficients, x._indeterminants)


@implements(numpy.add)
def add(x1, x2):
    x1, x2 = align.align_polynomials(x1, x2)
    collection = x1.todict()
    for exponent, coefficient in x2.todict().items():
        collection[exponent] = collection.get(exponent, False)+coefficient
    exponents = sorted(collection)
    coefficients = [collection[exponent] for exponent in exponents]
    return construct.polynomial_from_attributes(
        exponents, coefficients, x1._indeterminants)


@implements(numpy.any)
def any(a, **kwargs):
    coefficients = numpy.any(a.coefficients, 0).astype(bool)
    coefficients = numpy.any(coefficients, **kwargs)
    return coefficients


@implements(numpy.all)
def all(a, **kwargs):
    coefficients = numpy.any(a.coefficients, 0).astype(bool)
    coefficients = numpy.all(coefficients, **kwargs)
    return coefficients


@implements(numpy.array_repr)
def array_repr(a, **kwargs):
    del kwargs
    prefix = "polynomial("
    suffix = ")"
    array = export.to_array(a, as_type="str")
    return prefix + numpy.array2string(
        numpy.array(array),
        separator=", ",
        formatter={"all": str},
        prefix=prefix,
        suffix=suffix,
    ) + suffix


@implements(numpy.array_str)
def array_str(a, **kwargs):
    del kwargs
    prefix = ""
    suffix = ""
    array = export.to_array(a, as_type="str")
    return prefix + numpy.array2string(
        numpy.array(array),
        separator=" ",
        formatter={"all": str},
        prefix=prefix,
        suffix=suffix,
    ) + suffix


@implements(numpy.concatenate)
def concatenate(arrays, axis=0, out=None):
    """Wrapper for numpy.concatenate."""
    assert out is None, "'out' argument currently no supported"
    arrays = align.align_polynomial_indeterminants(*arrays)
    collections = [arg.todict() for arg in arrays]

    out = {}
    keys = {arg for collection in collections for arg in collection}
    for key in keys:
        values = [(collection[key] if key in collection
                   else numpy.zeros(array.shape, dtype=bool))
                  for collection, array in zip(collections, arrays)]
        out[key] = numpy.concatenate(values, axis=axis)

    exponents = sorted(out)
    coefficients = [out[exponent] for exponent in exponents]
    return construct.polynomial_from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        indeterminants=arrays[0].indeterminants,
    )


@implements(numpy.divide, numpy.true_divide)
def divide(x1, x2, **kwargs):
    assert not isinstance(x2, baseclass.ndpoly), "not supported"
    return multiply(x1, 1/numpy.asarray(x2), **kwargs)


@implements(numpy.equal)
def equal(x1, x2, **kwargs):
    x1, x2 = align.align_polynomials(x1, x2)
    out = numpy.ones(x1.shape, dtype=bool)
    collection = x1.todict()
    for exponent, coefficient in x2.todict().items():
        if exponent in collection:
            out &= collection.pop(exponent) == coefficient
        else:
            out &= coefficient == 0
    for _, coefficient in collection.items():
        out &= coefficient == 0
    return out


@implements(numpy.floor_divide)
def floor_divide(x1, x2, **kwargs):
    assert not isinstance(x2, baseclass.ndpoly), "not supported"
    return multiply(x1, 1/numpy.asarray(x2), **kwargs).astype(int)


@implements(numpy.multiply)
def multiply(x1, x2, **kwargs):
    assert kwargs.get("out", None) is None, "object read-only"
    x1, x2 = align.align_polynomials(x1, x2)
    exponents = (numpy.tile(x1.exponents, (len(x2.exponents), 1))+
                 numpy.repeat(x2.exponents, len(x1.exponents), 0))

    shape = (len(x2.coefficients),)+(1,)*len(x2.shape)
    coefficients = (numpy.tile(x1.coefficients, shape)*
                    numpy.repeat(x2.coefficients, len(x1.coefficients), 0))

    collection = {}
    for exponent, coefficient in zip(exponents.tolist(), coefficients):
        exponent = tuple(exponent)
        collection[exponent] = collection.get(exponent, False)+coefficient

    exponents = sorted(collection)
    coefficients = [collection[exponent] for exponent in exponents]

    return construct.polynomial_from_attributes(
        exponents, coefficients, x1._indeterminants)


@implements(numpy.negative)
def negative(x, out=None, **kwargs):
    x = x.copy()
    for exponent in x._exponents:
        x[exponent] = numpy.negative(x[exponent], **kwargs)
    return x


@implements(numpy.not_equal)
def not_equal(x1, x2, **kwargs):
    x1, x2 = align.align_polynomials(x1, x2)
    out = numpy.zeros(x1.shape, dtype=bool)
    collection = x1.todict()
    for exponent, coefficient in x2.todict().items():
        if exponent in collection:
            out |= collection.pop(exponent) != coefficient
        else:
            out |= coefficient != 0
    for _, coefficient in collection.items():
        out != coefficient != 0
    return out


@implements(numpy.positive)
def positive(x, **kwargs):
    x = x.copy()
    for exponent in x._exponents:
        x[exponent] = numpy.positive(x[exponent], **kwargs)
    return x


@implements(numpy.power)
def power(x1, x2, **kwargs):
    x1 = x1.copy()
    x2 = numpy.asarray(x2, dtype=int)

    if not x2.shape:
        out = construct.polynomial_from_attributes(
            [(0,)], [numpy.ones(x1.shape, dtype=x1._dtype)], x1._indeterminants[:1])
        for _ in range(x2.item()):
            out = multiply(out, x1, **kwargs)

    elif x1.shape:
        if x2.shape[-1] == 1:
            if x1.shape[-1] == 1:
                out = power(x1.T[0].T, x2.T[0].T).T[numpy.newaxis].T
            else:
                out = concatenate([power(x, x2.T[0])[numpy.newaxis] for x in x1.T], axis=0).T
        elif x1.shape[-1] == 1:
            out = concatenate([power(x1.T[0].T, x.T).T[numpy.newaxis] for x in x2.T], axis=0).T
        else:
            out = concatenate([power(x1_, x2_).T[numpy.newaxis] for x1_, x2_ in zip(x1.T, x2.T)], axis=0).T
    else:
        out = concatenate([power(x1, x.T).T[numpy.newaxis] for x in x2.T], axis=0).T
    return out


@implements(numpy.subtract)
def subtract(x1, x2, **kwargs):
    x1, x2 = align.align_polynomials(x1, x2)
    collection = x1.todict()
    for exponent, coefficient in x2.todict().items():
        collection[exponent] = collection.get(exponent, False)-coefficient
    exponents = sorted(collection)
    coefficients = [collection[exponent] for exponent in exponents]
    return construct.polynomial_from_attributes(
        exponents, coefficients, x1._indeterminants)


@implements(numpy.sum)
def sum(a, **kwargs):
    coefficients, kwargs = process_parameters(a, kwargs)
    coefficients = numpy.sum(coefficients, **kwargs)
    return construct.polynomial_from_attributes(
        a.exponents, coefficients, a._indeterminants)
