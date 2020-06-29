"""Return the floor of the input, element-wise."""
import numpy
import numpoly

from ..dispatch import implements, simple_dispatch


@implements(numpy.floor)
def floor(q0, out=None, where=True, **kwargs):
    r"""
    Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Args:
        x (numpoly.ndpoly):
            Input data.
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
        y (numpoly.ndpoly):
            The floor of each element in `x`. This is a scalar if `x` is
            a scalar.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.floor([-1.7*q0, q0-1.5, -0.2,
        ...                3.2+1.5*q0, 1.7, 2.0])
        polynomial([-2.0*q0, q0-2.0, -1.0, q0+3.0, 1.0, 2.0])

    """
    return simple_dispatch(
        numpy_func=numpy.floor,
        inputs=(q0,),
        out=out,
        where=where,
        **kwargs
    )
