"""Implementation wrapper."""
import numpy
import numpoly

ARRAY_FUNCTIONS = {}


def implements(*array_functions):
    """Register an __array_function__ implementation."""
    def decorator(numpoly_function):
        """Register function."""
        for func in array_functions:
            ARRAY_FUNCTIONS[func] = numpoly_function
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
    for key in inputs[0].keys:

        if out is None:
            tmp = numpy_func(*[poly[key] for poly in inputs], **kwargs)
            out = numpoly.ndpoly(
                exponents=inputs[0].exponents,
                shape=tmp.shape,
                names=inputs[0].indeterminants,
                dtype=tmp.dtype,
            )
            out[key] = tmp
        else:
            tmp = numpy_func(
                *[poly[key] for poly in inputs], out=out[key], **kwargs)

    if no_output:
        out = numpoly.clean_attributes(out)
    return out
