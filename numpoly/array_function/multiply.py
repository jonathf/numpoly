"""Multiply arguments element-wise."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.multiply)
def multiply(x1, x2, out=None, where=True, **kwargs):
    """
    Multiply arguments element-wise.

    Args:
        x1, x2 (numpoly.ndpoly):
            Input arrays to be multiplied. If ``x1.shape != x2.shape``, they
            must be broadcastable to a common shape (which becomes the shape of
            the output).
        out (Optional[numpy.ndarray]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (Optional[numpy.ndarray]):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        (numpoly.ndpoly):
            The product of `x1` and `x2`, element-wise. This is a scalar if
            both `x1` and `x2` are scalars.

    Examples:
        >>> poly = numpy.arange(9.0).reshape((3, 3))
        >>> q0q1q2 = numpoly.variable(3)
        >>> numpoly.multiply(poly, q0q1q2)
        polynomial([[0.0, q1, 2.0*q2],
                    [3.0*q0, 4.0*q1, 5.0*q2],
                    [6.0*q0, 7.0*q1, 8.0*q2]])

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    # x1, x2 = numpoly.broadcast_arrays(
    #     numpoly.polynomial(x1), numpoly.polynomial(x2))
    no_output = out is None
    if no_output:
        exponents = (numpy.tile(x1.exponents, (len(x2.exponents), 1))+
                     numpy.repeat(x2.exponents, len(x1.exponents), 0))
        out = numpoly.ndpoly(
            exponents=numpy.unique(exponents, axis=0),
            shape=x1.shape,
            names=x1.indeterminants,
            dtype=x1.dtype,
        )

    seen = set()
    for expon1, coeff1 in zip(x1.exponents, x1.coefficients):
        for expon2, coeff2 in zip(x2.exponents, x2.coefficients):
            key = (expon1+expon2+x1.KEY_OFFSET).flatten()
            key = key.view("U%d" % len(expon1)).item()
            if key in seen:
                out[key] += numpy.multiply(
                    coeff1, coeff2, where=where, **kwargs)
            else:
                numpy.multiply(
                    coeff1, coeff2, out=out[key], where=where, **kwargs)
            seen.add(key)
    if no_output:
        out = numpoly.clean_attributes(out)
    return out
