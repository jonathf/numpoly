from __future__ import division
import logging
import inspect
from functools import wraps

import numpy

from .align import align_polynomials
from . import construct, baseclass

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
    return construct.polynomial_from_attributes(x.exponents, coefficients)


@implements(numpy.add)
def add(x1, x2):
    x1, x2 = align_polynomials(x1, x2)
    collection = x1.todict()
    for exponent, coefficient in x2.todict().items():
        collection[exponent] = collection.get(exponent, False)+coefficient
    exponents = sorted(collection)
    coefficients = [collection[exponent] for exponent in exponents]
    return construct.polynomial_from_attributes(exponents, coefficients)


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
    from .string import construct_string_array
    prefix = "polynomial("
    suffix = ")"
    array = construct_string_array(
        a, sep="*", power="**", **kwargs)
    return prefix + numpy.array2string(
        numpy.array(array),
        separator=", ",
        formatter={"all": str},
        prefix=prefix,
        suffix=suffix,
    ) + suffix


@implements(numpy.array_str)
def array_str(a, **kwargs):
    from .string import construct_string_array
    prefix = ""
    suffix = ""
    array = construct_string_array(
        a, sep="", power="^", **kwargs)
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
    arrays = [construct.polynomial(arg) for arg in arrays]
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
    return construct.polynomial_from_attributes(exponents, coefficients)


@implements(numpy.divide, numpy.true_divide)
def divide(x1, x2, **kwargs):
    assert not isinstance(x2, baseclass.ndpoly), "not supported"
    return multiply(x1, 1/numpy.asarray(x2), **kwargs)


@implements(numpy.equal)
def equal(x1, x2, **kwargs):
    x1, x2 = align_polynomials(x1, x2)
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
    x1, x2 = align_polynomials(x1, x2)
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

    return construct.polynomial_from_attributes(exponents, coefficients)


@implements(numpy.negative)
def negative(x, out=None, **kwargs):
    x = x.copy()
    for key in x._keys:
        x[key] = numpy.negative(x[key], **kwargs)
    return x


@implements(numpy.not_equal)
def not_equal(x1, x2, **kwargs):
    x1, x2 = align_polynomials(x1, x2)
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
    for key in x._keys:
        x[key] = numpy.positive(x[key], **kwargs)
    return x


@implements(numpy.power)
def power(x1, x2, **kwargs):
    x1 = x1.copy()
    x2 = int(x2)
    assert x2 >= 0, "only positive integers allowed"
    out = construct.polynomial_from_attributes([(0,)], [numpy.ones(x1.shape, dtype=x1._dtype)])
    for _ in range(x2):
        out = multiply(out, x1, **kwargs)
    return out
    # x1 = x1.copy()
    # x2 = numpy.asarray(x2, dtype=int)
    # assert numpy.all(x2 >= 0), "only positive integers allowed"
    # out = construct.polynomial_from_attributes([(0,)], [numpy.ones(x1.shape, dtype=x1._dtype)])
    # while numpy.any(x2):
    #     indices = x2 >= 0
    #     out[indices] = multiply(out[indices], x1[indices], **kwargs)
    #     x2 -= 1
    # return out


@implements(numpy.subtract)
def subtract(x1, x2, **kwargs):
    x1, x2 = align_polynomials(x1, x2)
    collection = x1.todict()
    for exponent, coefficient in x2.todict().items():
        collection[exponent] = collection.get(exponent, False)-coefficient
    exponents = sorted(collection)
    coefficients = [collection[exponent] for exponent in exponents]
    return construct.polynomial_from_attributes(exponents, coefficients)


@implements(numpy.sum)
def sum(a, **kwargs):
    coefficients, kwargs = process_parameters(a, kwargs)
    coefficients = numpy.sum(coefficients, **kwargs)
    return construct.polynomial_from_attributes(a.exponents, coefficients)
