"""Return a new array of given shape and type, filled with ones."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.ones)
def ones(shape, dtype=float, order="C"):
    """
    Return a new array of given shape and type, filled with ones.

    Args:
        shape : int or tuple of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is
            `numpy.float64`.
        order : {'C', 'F'}, optional, default: 'C'
            Whether to store multi-dimensional data in row-major
            (C-style) or column-major (Fortran-style) order in
            memory.

    Returns:
        (numpoly.ndpoly):
            Array of ones with the given shape, dtype, and order.

    Examples:
        >>> numpoly.ones(5)
        polynomial([1.0, 1.0, 1.0, 1.0, 1.0])

    """
    return numpoly.polynomial(numpy.ones(shape, dtype=dtype, order=order))
