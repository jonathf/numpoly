"""Return a string representation of the data in an array."""
import numpy
import numpoly

from ..dispatch import implements
from .array_repr import to_string


@implements(numpy.array_str)
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Args:
        a (numpoly.ndpoly):
            Input array.
        max_line_width (Optional[int]):
            Inserts newlines if text is longer than `max_line_width`. Defaults
            to ``numpy.get_printoptions()['linewidth']``.
        precision (Optional[int]):
            Floating point precision. Defaults to
            ``numpy.get_printoptions()['precision']``.
        suppress_small (Optional[bool]):
            Represent numbers "very close" to zero as zero; default is False.
            Very close is defined by precision: if the precision is 8, e.g.,
            numbers smaller (in absolute value) than 5e-9 are represented as
            zero. Defaults to ``numpy.get_printoptions()['suppress']``.

    Returns:
        (str):
            The string representation of an array.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.array_str(numpoly.polynomial([1, q0]))
        '[1 q0]'
        >>> numpoly.array_str(numpoly.polynomial([]))
        '[]'
        >>> numpoly.array_str(
        ...     numpoly.polynomial([1e-6, 4e-7*q0, 2*q0, 3]),
        ...     precision=4,
        ...     suppress_small=True,
        ... )
        '[0.0 0.0 2.0*q0 3.0]'

    """
    a = numpoly.aspolynomial(a)
    a = to_string(a, precision=precision, suppress_small=suppress_small)
    return numpy.array2string(
        numpy.array(a),
        max_line_width=max_line_width,
        separator=" ",
        formatter={"all": str},
        prefix="",
        suffix="",
    )
