"""Calculate the absolute value element-wise."""
import numpy
import numpoly

from .common import implements, simple_dispatch


@implements(numpy.abs, numpy.absolute)
def absolute(x, out=None, where=True, **kwargs):
    r"""
    Calculate the absolute value element-wise.

    Args:
        x (numpoly.ndpoly):
            Input array.
        out (Optional[numpy.ndarray]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (Union[bool, numpy.ndarray]):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        absolute (numpoly.ndpoly):
            An ndarray containing the absolute value of. each element in `x`.
            For complex input, ``a + ib``, the absolute value is
            :math:`\sqrt{a^2+b^2}`. This is a scalar if `x` is a scalar.

    Examples:
        >>> x = numpoly.symbols("x")
        >>> poly = numpoly.polynomial([-1.2, 1.2, -2.3*x, 2.3*x])
        >>> poly
        polynomial([-1.2, 1.2, -2.3*x, 2.3*x])
        >>> numpoly.absolute(poly)
        polynomial([1.2, 1.2, 2.3*x, 2.3*x])
        >>> poly = numpoly.polynomial([x, 1j*x, (3+4j)*x])
        >>> poly
        polynomial([x, 1j*x, (3+4j)*x])
        >>> numpoly.absolute(poly)
        polynomial([x, x, 5.0*x])

    """
    return simple_dispatch(
        numpy_func=numpy.absolute,
        inputs=(x,),
        out=out,
        where=where,
        **kwargs
    )
