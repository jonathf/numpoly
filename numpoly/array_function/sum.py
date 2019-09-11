"""Sum of array elements over a given axis."""
import numpy
import numpoly

from .implements import implements


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
        >>> x, y, z = xyz = numpoly.symbols("x y z")
        >>> numpoly.sum(xyz)
        polynomial(x+y+z)
        >>> numpoly.sum([[1, x], [y, z]])
        polynomial(1+z+y+x)
        >>> numpoly.sum([[1, x], [y, z]], axis=0)
        polynomial([1+y, z+x])

    """
    a = numpoly.aspolynomial(a)
    no_output = out is None
    if no_output:
        if axis is not None:
            axes = [axis] if isinstance(axis, (int, numpy.generic)) else axis
            axes = [(ax if ax >= 0 else len(a.shape)+ax) for ax in axes]
            if keepdims:
                shape = [(1 if idx in axes else dim)
                         for idx, dim in enumerate(a.shape)]
            else:
                shape = [dim for idx, dim in enumerate(a.shape)
                         if idx not in axes]
        elif keepdims:
            shape = (1,)*len(a.shape)
        else:
            shape = ()
        out = numpoly.ndpoly(
            exponents=a.exponents,
            shape=shape,
            indeterminants=a.indeterminants,
            dtype=a.dtype,
        )

    for key in a.keys:
        numpy.sum(a[key], axis=axis, dtype=dtype, out=out[key],
                  keepdims=keepdims, **kwargs)
    if no_output:
        out = numpoly.clean_attributes(out)
    return out
