"""Implementation wrapper."""
from typing import Any, Callable, Optional, Sequence
import numpoly

from .baseclass import ndpoly

FUNCTION_COLLECTION = {}
UFUNC_COLLECTION = {}


def implements_function(*array_functions: Callable) -> Callable:
    """Register __array_function__."""
    def decorator(numpoly_function: Callable) -> Callable:
        """Register function."""
        for func in array_functions:
            assert func not in FUNCTION_COLLECTION, f"{func} already implemented"
            FUNCTION_COLLECTION[func] = numpoly_function
        return numpoly_function
    return decorator


def implements_ufunc(*array_methods: Callable) -> Callable:
    """Register __array_ufunc__."""
    def decorator(numpoly_function: Callable) -> Callable:
        """Register function."""
        for func in array_methods:
            assert func not in UFUNC_COLLECTION, f"{func} already implemented"
            UFUNC_COLLECTION[func] = numpoly_function
        return numpoly_function
    return decorator


def implements(*array_functions: Callable) -> Callable:
    """Register __array_function__ and __array_ufunc__."""
    def decorator(numpoly_function: Callable) -> Callable:
        """Register function."""
        for func in array_functions:
            assert func not in FUNCTION_COLLECTION, "{func} already implemented"
            FUNCTION_COLLECTION[func] = numpoly_function
            assert func not in UFUNC_COLLECTION, "{func} already implemented"
            UFUNC_COLLECTION[func] = numpoly_function
        return numpoly_function

    return decorator


def simple_dispatch(
        numpy_func: Callable,
        inputs: Sequence[Any],
        out: Optional[ndpoly] = None,
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
    no_output = out is None
    keys = numpoly.aspolynomial(inputs[0] if no_output else out).keys
    for key in keys:

        if out is None:
            tmp = numpy_func(*[poly[key] for poly in inputs], **kwargs)
            out = numpoly.ndpoly(
                exponents=inputs[0].exponents,
                shape=tmp.shape,
                names=inputs[0].indeterminants,
                dtype=tmp.dtype,
            )
            out[key] = tmp

        elif no_output:
            out[key] = numpy_func(*[poly[key] for poly in inputs], **kwargs)

        else:
            tmp = numpy_func(
                *[poly[key] for poly in inputs], out=out[key], **kwargs)

    if no_output:
        out = numpoly.clean_attributes(out)
    return numpoly.aspolynomial(out)
