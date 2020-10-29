"""Return a new array of given shape and type, filled with `fill_value`."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.full)
def full(shape, fill_value, dtype=None, order="C"):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Args:
        shape (int, Sequence[int]):
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        fill_value (numpoly.ndpoly):
            Fill value. Must be broadcast compatible with `shape`.
        dtype : data-type, optional
            The desired data-type for the array  The default, None, means
            inherit from `fill_value`.
        order : {'C', 'F'}, optional
            Whether to store multidimensional data in C- or Fortran-contiguous
            (row- or column-wise) order in memory.

    Returns:
        out : ndarray
            Array of `fill_value` with the given shape, dtype, and order.

    Examples:
        >>> numpoly.full((2, 4), 4)
        polynomial([[4, 4, 4, 4],
                    [4, 4, 4, 4]])
        >>> q0 = numpoly.variable()
        >>> numpoly.full(3, q0**2-1)
        polynomial([q0**2-1, q0**2-1, q0**2-1])

    """
    fill_value = numpoly.aspolynomial(fill_value)
    if dtype is None:
        dtype = fill_value.dtype
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
