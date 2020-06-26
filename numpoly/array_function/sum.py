"""Sum of array elements over a given axis."""
import numpy
import numpoly

from ..dispatch import implements, simple_dispatch


@implements(numpy.sum)
def sum(a, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
    """
    Sum of array elements over a given axis.

    Args:
        a (numpoly.ndpoly):
            Elements to sum.
        axis (Union[None, int, Tuple[int, ...]]):
            Axis or axes along which a sum is performed. The default,
            axis=None, will sum all of the elements of the input array. If axis
            is negative it counts from the last to the first axis. If axis is
            a tuple of ints, a sum is performed on all of the axes specified in
            the tuple instead of a single axis or all the axes as before.
        dtype (Optional[numpy.ndtype]):
            The type of the returned array and of the accumulator in which the
            elements are summed.  The dtype of `a` is used by default unless
            `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.
        out (Optional[numpy.ndarray]):
            Alternative output array in which to place the result. It must have
            the same shape as the expected output, but the type of the output
            values will be cast if necessary.
        keepdims (Optional[bool]):
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
        initial (Union[None, int, float, complex]):
            Starting value for the sum.
        where (Optional[numpy.ndarray]):
            Elements to include in the sum.

    Returns:
        (numpoly.ndpoly):
            An array with the same shape as `a`, with the specified axis
            removed. If `a` is a 0-d array, or if `axis` is None, a scalar is
            returned. If an output array is specified, a reference to `out` is
            returned.

    Examples:
        >>> q0, q1, q2 = numpoly.variable(3)
        >>> poly = numpoly.polynomial([[1, q0], [q1, q2]])
        >>> numpoly.sum(poly)
        polynomial(q2+q1+q0+1)
        >>> numpoly.sum(poly, axis=0)
        polynomial([q1+1, q2+q0])

    """
    return simple_dispatch(
        numpy_func=numpy.sum,
        inputs=(a,),
        out=out,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        **kwargs
    )
