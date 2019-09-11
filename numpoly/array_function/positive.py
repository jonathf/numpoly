"""Numerical positive, element-wise."""
import numpy
import numpoly

from .implements import implements


@implements(numpy.positive)  # pylint: disable=no-member
def positive(x, out=None, where=True, **kwargs):
    """
    Numerical positive, element-wise.

    Args:
        x (numpoly.ndpoly):
            Input array.
        out (Optional[numpoly.ndpoly]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where : array_like, optional
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        **kwargs
            Keyword args passed to numpy.ufunc.

    Returns:
        y (numpoly.ndpoly):
            Returned array or scalar: `y = +x`. This is a scalar if `x` is
            a scalar.

    Examples:
        >>> x = numpoly.symbols("x")
        >>> numpoly.positive([-0, 0, -x, x])
        polynomial([0, 0, -x, x])

    """
    x = numpoly.aspolynomial(x)
    if out is None:
        out = numpoly.ndpoly(
            exponents=x.exponents,
            shape=x.shape,
            indeterminants=x.indeterminants,
            dtype=x.dtype,
        )
    for key in x.keys:
        numpy.positive(  # pylint: disable=no-member
            x[key], out=out[key], where=where, **kwargs)
    return out
