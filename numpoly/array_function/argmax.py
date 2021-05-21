"""Return the indices of the maximum values along an axis."""
from __future__ import annotations
from typing import Any, Optional

import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.argmax)
def argmax(
    a: PolyLike,
    axis: Optional[int] = None,
    out: Optional[numpy.ndarray] = None,
) -> Any:
    """
    Return the indices of the maximum values along an axis.

    As polynomials are not inherently sortable, values are sorted using the
    highest `lexicographical` ordering. Between the values that have the same
    highest ordering, the elements are sorted using the coefficients. This also
    ensures that the method behaves as expected with ``numpy.ndarray``.

    Args:
        a:
            Input array.
        axis:
            By default, the index is into the flattened array, otherwise along
            the specified axis.
        out:
            If provided, the result will be inserted into this array. It should
            be of the appropriate shape and dtype.

    Returns:
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    Notes:
        In case of multiple occurrences of the maximum values, the
        indices corresponding to the first occurrence are returned.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.argmax([13, 7])
        0
        >>> numpoly.argmax([1, q0, q0**2, q1])
        2
        >>> numpoly.argmax([1, q0, q1])
        2
        >>> numpoly.argmax([[3*q0**2, q0**2],
        ...                 [2*q0**2, 4*q0**2]], axis=0)
        array([0, 1])

    """
    a = numpoly.aspolynomial(a)
    options = numpoly.get_options()
    proxy = numpoly.sortable_proxy(
        a, graded=options["sort_graded"], reverse=options["sort_reverse"])
    return numpy.argmax(proxy, axis=axis, out=out)
