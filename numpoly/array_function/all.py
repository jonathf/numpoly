"""Test whether all array elements along a given axis evaluate to True."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.all)
def all(a, axis=None, out=None, keepdims=False, **kwargs):
    """
    Test whether all array elements along a given axis evaluate to True.

    Args:
        a (numpoly.ndpoly):
            Input array or object that can be converted to an array.
        axis (Union[int, Tuple[int], None]):
            Axis or axes along which a logical AND reduction is performed. The
            default (`axis` = `None`) is to perform a logical AND over all the
            dimensions of the input array. `axis` may be negative, in which
            case it counts from the last to the first axis. If this is a tuple
            of ints, a reduction is performed on multiple axes, instead of
            a single axis or all the axes as before.
        out (Optional[numpy.ndarray]):
            Alternate output array in which to place the result. It must have
            the same shape as the expected output and its type is preserved
            (e.g., if ``dtype(out)`` is float, the result will consist of 0.0's
            and 1.0's).
        keepdims (Optional[bool]):
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

    Returns:
        any (Union[bool, numpy.ndarray]):
            A new boolean or array is returned unless `out` is specified, in
            which case a reference to `out` is returned.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.all(q0)
        True
        >>> numpoly.all(0*q0)
        False
        >>> numpoly.all([1, q0, 0])
        False
        >>> numpoly.all([[True*q0, False], [True, True]], axis=0)
        array([ True, False])

    """
    a = numpoly.aspolynomial(a)
    coefficients = numpy.any(a.coefficients, 0).astype(bool)
    out = numpy.all(
        coefficients, axis=axis, out=out, keepdims=keepdims)
    return out
