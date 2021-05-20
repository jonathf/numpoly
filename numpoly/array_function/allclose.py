"""Return True if two arrays are element-wise equal within a tolerance."""
from __future__ import annotations

import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.allclose)
def allclose(
    a: PolyLike,
    b: PolyLike,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> bool:
    """
    Return True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    If either array contains one or more NaNs, False is returned.
    Infs are treated as equal if they are in the same place and of the same
    sign in both arrays.

    Args:
        a, b:
            Input arrays to compare.
        rtol:
            The relative tolerance parameter (see Notes).
        atol:
            The absolute tolerance parameter (see Notes).
        equal_nan:
            Whether to compare NaN's as equal.  If True, NaN's in `a` will be
            considered equal to NaN's in `b` in the output array.

    Returns:
        Returns True if the two arrays are equal within the given tolerance;
        False otherwise.

    Notes:
        If the following equation is element-wise True, then allclose returns
        True.

        absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

        The above equation is not symmetric in `a` and `b`, so that
        ``allclose(a, b)`` might be different from ``allclose(b, a)`` in some
        rare cases.

        The comparison of `a` and `b` uses standard broadcasting, which means
        that `a` and `b` need not have the same shape in order for
        ``allclose(a, b)`` to evaluate to True.  The same is true for `equal`
        but not `array_equal`.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.allclose([1e9*q0, 1e-7], [1.00001e9*q0, 1e-8])
        False
        >>> numpoly.allclose([1e9*q0, 1e-8], [1.00001e9*q0, 1e-9])
        True
        >>> numpoly.allclose([1e9*q0, 1e-8], [1.00001e9*q1, 1e-9])
        False
        >>> numpoly.allclose([q0, numpy.nan],
        ...                  [q0, numpy.nan], equal_nan=True)
        True

    """
    a, b = numpoly.align_polynomials(a, b)
    for coeff1, coeff2 in zip(a.coefficients, b.coefficients):
        if not numpy.allclose(
                coeff1, coeff2, atol=atol, rtol=rtol, equal_nan=equal_nan):
            return False
    return True
