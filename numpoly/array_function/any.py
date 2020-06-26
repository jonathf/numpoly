"""Test whether any array element along a given axis evaluates to True."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.any)
def any(a, axis=None, out=None, keepdims=False, **kwargs):
    """
    Test whether any array element along a given axis evaluates to True.

    Returns single boolean unless `axis` is not ``None``

    Args:
        a (numpoly.ndpoly):
            Input array or object that can be converted to an array.
        axis (Union[int, Tuple[int], None]):
            Axis or axes along which a logical OR reduction is performed. The
            default (`axis` = `None`) is to perform a logical OR over all the
            dimensions of the input array. `axis` may be negative, in which
            case it counts from the last to the first axis. If this is a tuple
            of ints, a reduction is performed on multiple axes, instead of
            a single axis or all the axes as before.
        out (Optional[numpy.ndarray]):
            Alternate output array in which to place the result.  It must have
            the same shape as the expected output and its type is preserved
            (e.g., if it is of type float, then it will remain so, returning
            1.0 for True and 0.0 for False, regardless of the type of `a`).
        keepdims (Optional[bool]):
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.

    Returns:
        any (Union[bool, numpy.ndarray]):
            A new boolean or `ndarray` is returned unless `out` is specified,
            in which case a reference to `out` is returned.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.any(q0)
        True
        >>> numpoly.any(0*q0)
        False
        >>> numpoly.any([1, q0, 0])
        True
        >>> numpoly.any([[True*q0, False], [True, True]], axis=0)
        array([ True,  True])

    """
    a = numpoly.aspolynomial(a)
    coefficients = numpy.any(a.coefficients, 0).astype(bool)
    out = numpy.any(
        coefficients, axis=axis, out=out, keepdims=keepdims)
    return out
