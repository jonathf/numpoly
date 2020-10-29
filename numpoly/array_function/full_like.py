"""Return a full array with the same shape and type as a given array."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.full_like)
def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    """
    Return a full array with the same shape and type as a given array.

    Args:
        a (numpoly.ndpoly, numpy.ndarray):
            The shape and data-type of `a` define these same attributes of
            the returned array.
        fill_value (numpoly.ndpoly):
            Fill value. Must be broadcast compatible with shape.
        dtype (Optional[numpy.dtype]):
            Overrides the data type of the result.
        order (str):
            Overrides the memory layout of the result. 'C' means C-order,
            'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
            'C' otherwise. 'K' means match the layout of `a` as closely
            as possible.
        subok (bool):
            If True, then the newly created array will use the sub-class
            type of 'a', otherwise it will be a base-class array. Defaults
            to True.
        shape (Optional[Sequence[int]]):
            Overrides the shape of the result. If order='K' and the number of
            dimensions is unchanged, will try to keep order, otherwise,
            order='C' is implied.

    Returns:
        (numpoly.ndpoly):
            Array of `fill_value` with the same shape and type as `a`.

    Examples:
        >>> poly = numpoly.monomial(3)
        >>> poly
        polynomial([1, q0, q0**2])
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.full_like(poly, q1-1.)
        polynomial([q1-1, q1-1, q1-1])

    """
    del subok
    if not isinstance(a, numpy.ndarray):
        a = numpoly.polynomial(a)
    fill_value = numpoly.aspolynomial(fill_value)
    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    if order in ("A", "K"):
        order = "F" if a.flags["F_CONTIGUOUS"] else "C"
    out = numpoly.ndpoly(
        exponents=fill_value.exponents,
        shape=shape,
        names=fill_value.indeterminants,
        dtype=dtype,
        order=order,
    )
    for key in fill_value.keys:
        out[key] = fill_value[key]
    return out
