"""Compute the arithmetic mean along the specified axis."""
import numpy
import numpoly

from .common import implements, simple_dispatch


@implements(numpy.mean)
def mean(a, axis=None, dtype=None, out=None, **kwargs):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Args:
        a (numpoly.ndpoly):
            Array containing numbers whose mean is desired. If `a` is not an
            array, a conversion is attempted.
        axis (Optional[numpy.ndarray]):
            Axis or axes along which the means are computed. The default is to
            compute the mean of the flattened array. If this is a tuple of
            ints, a mean is performed over multiple axes, instead of a single
            axis or all the axes as before.
        dtype (Optional[numpy.dtype]):
            Type to use in computing the mean.  For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.
        out (Optional[numpy.ndpoly]):
            Alternate output array in which to place the result.  The default
            is ``None``; if provided, it must have the same shape as the
            expected output, but the type will be cast if necessary.
        keepdims (Optional[bool]):
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        (numpy.ndpoly):
            If `out=None`, returns a new array containing the mean values,
            otherwise a reference to the output array is returned.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> a = numpoly.polynomial([[1, 2*x], [3*y+x, 4]])
        >>> numpoly.mean(a)
        polynomial(1.25+0.75*x+0.75*y)
        >>> numpoly.mean(a, axis=0)
        polynomial([0.5+0.5*x+1.5*y, 2.0+x])
        >>> numpoly.mean(a, axis=1)
        polynomial([0.5+x, 2.0+0.5*x+1.5*y])

    """
    return simple_dispatch(
        numpy_func=numpy.mean,
        inputs=(a,),
        out=out,
        axis=axis,
        dtype=dtype,
        **kwargs
    )
