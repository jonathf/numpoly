"""Difference between consecutive elements of an array."""
from __future__ import annotations
from typing import Optional

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.ediff1d)
def ediff1d(
    ary: PolyLike,
    to_end: Optional[PolyLike] = None,
    to_begin: Optional[PolyLike] = None,
) -> ndpoly:
    """
    Difference between consecutive elements of an array.

    Args:
        ary:
            If necessary, will be flattened before the differences are taken.
        to_end:
            Polynomial(s) to append at the end of the returned differences.
        to_begin:
            Polynomial(s) to prepend at the beginning of the returned
            differences.

    Returns:
        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

    Examples:
        >>> poly = numpoly.monomial(4)
        >>> poly
        polynomial([1, q0, q0**2, q0**3])
        >>> numpoly.ediff1d(poly)
        polynomial([q0-1, q0**2-q0, q0**3-q0**2])
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.ediff1d(poly, to_begin=q0, to_end=[1, q1])
        polynomial([q0, q0-1, q0**2-q0, q0**3-q0**2, 1, q1])

    """
    ary = numpoly.aspolynomial(ary).ravel()
    arys_ = [ary[1:]-ary[:-1]]
    if to_end is not None:
        arys_.append(numpoly.aspolynomial(to_end).ravel())
    if to_begin is not None:
        arys_.insert(0, numpoly.aspolynomial(to_begin).ravel())
    arys = tuple(numpoly.aspolynomial(ary) for ary in arys_)
    if len(arys) > 1:
        arys = numpoly.align_exponents(*arys)

    out = numpoly.ndpoly(
        exponents=arys[0].exponents,
        shape=(sum([ary.size for ary in arys]),),
        names=arys[0].names,
        dtype=ary[0].dtype,
    )

    idx = 0
    for ary in arys:
        for key in ary.keys:
            out.values[key][idx:idx+ary.size] = ary.values[key]
        idx += ary.size

    return out
