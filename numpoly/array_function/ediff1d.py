"""Difference between consecutive elements of an array."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.ediff1d)
def ediff1d(ary, to_end=None, to_begin=None):
    """
    Difference between consecutive elements of an array.

    Args:
        ary (numpoly.ndpoly):
            If necessary, will be flattened before the differences are taken.
        to_end (Optional[numpoly.ndpoly]):
            Polynomial(s) to append at the end of the returned differences.
        to_begin (Optional[numpoly.ndpoly]):
            Polynomial(s) to prepend at the beginning of the returned
            differences.

    Returns:
        (numpoly.ndpoly):
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
    arys = [ary[1:]-ary[:-1]]
    if to_end is not None:
        arys.append(numpoly.aspolynomial(to_end).ravel())
    if to_begin is not None:
        arys.insert(0, numpoly.aspolynomial(to_begin).ravel())
    if len(arys) > 1:
        arys = numpoly.align_dtype(*arys)
        arys = numpoly.align_exponents(*arys)
        arys = numpoly.align_indeterminants(*arys)

    out = numpoly.ndpoly(
        exponents=arys[0].exponents,
        shape=(sum([ary.size for ary in arys]),),
        names=arys[0].names,
        dtype=ary[0].dtype,
    )

    idx = 0
    for ary in arys:
        for key in ary.keys:
            out[key][idx:idx+ary.size] = ary[key]
        idx += ary.size

    return out
