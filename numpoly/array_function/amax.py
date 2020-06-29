"""Return the maximum of an array or maximum along an axis."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.amax)
def amax(a, axis=None, out=None, **kwargs):
    """
    Return the maximum of an array or maximum along an axis.

    Args:
        a (numpoly.ndpoly):
            Input data.
        axis (int, Tuple[int], None):
            Axis or axes along which to operate.  By default, flattened input
            is used. If this is a tuple of ints, the maximum is selected over
            multiple axes, instead of a single axis or all the axes as before.
        out (Optional[numpoly.ndpoly]):
            Alternative output array in which to place the result. Must be of
            the same shape and buffer length as the expected output.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.

            If the default value is passed, then `keepdims` will not be passed
            through to the `amax` method of sub-classes of `ndarray`, however
            any non-default value will be.  If the sub-class' method does not
            implement `keepdims` any exceptions will be raised.
        initial : scalar, optional
            The minimum value of an output element. Must be present to allow
            computation on empty slice.
        where : array_like of bool, optional
            Elements to compare for the maximum.

    Returns:
        (numpy.ndarray):
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is an array of dimension
            ``a.ndim-1``.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.amax([13, 7])
        polynomial(13)
        >>> numpoly.amax([1, q0, q0**2, q1])
        polynomial(q0**2)
        >>> numpoly.amax([1, q0, q1])
        polynomial(q1)
        >>> numpoly.amax([[3*q0**2, 5*q0**2],
        ...               [2*q0**2, 4*q0**2]], axis=0)
        polynomial([3*q0**2, 5*q0**2])

    """
    del out
    a = numpoly.aspolynomial(a)
    options = numpoly.get_options()
    proxy = numpoly.sortable_proxy(
        a, graded=options["sort_graded"], reverse=options["sort_reverse"])
    indices = numpy.amax(proxy, axis=axis, **kwargs)
    out = a[numpy.isin(proxy, indices)]
    out = out[numpy.argsort(indices.ravel())]
    return out.reshape(indices.shape)
