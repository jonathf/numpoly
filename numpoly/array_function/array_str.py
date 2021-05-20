"""Return a string representation of the data in an array."""
from __future__ import annotations
from typing import Optional

import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements
from .array_repr import to_string


@implements(numpy.array_str)
def array_str(
        a: PolyLike,
        max_line_width: Optional[int] = None,
        precision: Optional[float] = None,
        suppress_small: Optional[bool] = None,
) -> str:
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Args:
        a:
            Input array.
        max_line_width:
            Inserts newlines if text is longer than `max_line_width`. Defaults
            to ``numpy.get_printoptions()['linewidth']``.
        precision:
            Floating point precision. Defaults to
            ``numpy.get_printoptions()['precision']``.
        suppress_small:
            Represent numbers "very close" to zero as zero; default is False.
            Very close is defined by precision: if the precision is 8, e.g.,
            numbers smaller (in absolute value) than 5e-9 are represented as
            zero. Defaults to ``numpy.get_printoptions()['suppress']``.

    Returns:
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
