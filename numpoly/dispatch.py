"""Implementation wrapper."""
import numpy
import numpoly

FUNCTION_COLLECTION = {}
UFUNC_COLLECTION = {}


def implements_function(*array_functions):
    """Register __array_function__."""
    def decorator(numpoly_function):
        """Register function."""
        for func in array_functions:
            assert func not in FUNCTION_COLLECTION, "%s already implemented" % func
            FUNCTION_COLLECTION[func] = numpoly_function
        return numpoly_function
    return decorator


def implements_ufunc(*array_methods):
    """Register __array_ufunc__."""
    def decorator(numpoly_function):
        """Register function."""
        for func in array_methods:
            assert func not in UFUNC_COLLECTION, "%s already implemented" % func
            UFUNC_COLLECTION[func] = numpoly_function
        return numpoly_function
    return decorator


def implements(*array_functions):
    """Register __array_function__ and __array_ufunc__."""
    def decorator(numpoly_function):
        """Register function."""
        for func in array_functions:
            assert func not in FUNCTION_COLLECTION, "%s already implemented" % func
            FUNCTION_COLLECTION[func] = numpoly_function
            assert func not in UFUNC_COLLECTION, "%s already implemented" % func
            UFUNC_COLLECTION[func] = numpoly_function
        return numpoly_function

    return decorator


def simple_dispatch(
        numpy_func,
        inputs,
        out=None,
        **kwargs
):
    """
    Dispatch function between numpy and numpoly.

    Assumes that evaluation can be performed on the coefficients alone and that
    there are no change to the polynomials.

    Args:
        numpy_func (Callable):
            The numpy function to evaluate `inputs` on.
        inputs (Iterable[numpoly.ndpoly]):
            One or more input arrays.
        out (Optional[numpy.ndarray]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        kwargs:
            Keyword args passed to `numpy_func`.

    Returns:
        (numpoly.ndpoly):
            Polynomial, where the coefficients from `input` are passed to
            `numpy_func` to create the output coefficients.

    """
    inputs = numpoly.align_polynomials(*inputs)
    no_output = out is None
    keys = inputs[0].keys if no_output else out.keys
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
    return out
