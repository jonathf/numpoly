"""Implementation wrapper."""
from typing import Any, Callable, Optional, Sequence, Tuple

import numpoly

from .baseclass import ndpoly

FUNCTION_COLLECTION = {}
UFUNC_COLLECTION = {}


def implements_function(*array_functions: Callable) -> Callable:
    """Register __array_function__."""
    def decorator(numpoly_function: Callable) -> Callable:
        """Register function."""
        for func in array_functions:
            assert func not in FUNCTION_COLLECTION, (
                f"{func} already implemented")
            FUNCTION_COLLECTION[func] = numpoly_function
        return numpoly_function
    return decorator


def implements_ufunc(*array_methods: Callable) -> Callable:
    """Register __array_ufunc__."""
    def decorator(numpoly_function: Callable) -> Callable:
        """Register function."""
        for func in array_methods:
            assert func not in UFUNC_COLLECTION, (
                f"{func} already implemented")
            UFUNC_COLLECTION[func] = numpoly_function
        return numpoly_function
    return decorator


def implements(*array_functions: Callable) -> Callable:
    """Register __array_function__ and __array_ufunc__."""
    def decorator(numpoly_function: Callable) -> Callable:
        """Register function."""
        for func in array_functions:
            assert func not in FUNCTION_COLLECTION, (
                "{func} already implemented")
            FUNCTION_COLLECTION[func] = numpoly_function
            assert func not in UFUNC_COLLECTION, "{func} already implemented"
            UFUNC_COLLECTION[func] = numpoly_function
        return numpoly_function

    return decorator


def simple_dispatch(
        numpy_func: Callable,
        inputs: Sequence[Any],
        out: Optional[Tuple[ndpoly, ...]] = None,
        **kwargs: Any
) -> ndpoly:
    """
    Dispatch function between numpy and numpoly.

    Assumes that evaluation can be performed on the coefficients alone and that
    there are no change to the polynomials.

    Args:
        numpy_func:
            The numpy function to evaluate `inputs` on.
        inputs:
            One or more input arrays.
        out:
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        kwargs:
            Keyword args passed to `numpy_func`.

    Returns:
        Polynomial, where the coefficients from `input` are passed to
        `numpy_func` to create the output coefficients.

    """
    inputs = numpoly.align_polynomials(*inputs)
    keys = (inputs[0] if out is None else numpoly.aspolynomial(out[0])).keys

    tmp = numpy_func(*[poly.values[keys[0]]for poly in inputs], **kwargs)
    if out is None:
        out_ = numpoly.ndpoly(
            exponents=inputs[0].exponents,
            shape=tmp.shape,
            names=inputs[0].indeterminants,
            dtype=tmp.dtype,
        )
    else:
        assert len(out) == 1
        out_ = out[0]
    out_.values[keys[0]] = tmp

    for key in keys[1:]:
        out_.values[key] = numpy_func(
            *[poly.values[key] for poly in inputs], **kwargs)

    if out is None:
        out_ = numpoly.clean_attributes(out_)
    return numpoly.aspolynomial(out_)
