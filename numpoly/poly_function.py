import numpy
from .construct import polynomial_from_attributes


def variable(dimensions=1, dtype="i8"):
    """
    Simple constructor to create single variables to create polynomials.

    Args:
        dimensions (int):
            Number of dimensions in the array.
        dtype:

    Returns:
        (numpoly.ndpoly):
            Polynomial array with unit components in each dimension.

    Examples:
        >>> print(numpoly.variable())
        q0
        >>> print(numpoly.variable(3))
        [q0 q1 q2]
    """
    keys = numpy.eye(dimensions, dtype=dtype)
    values = numpy.eye(dimensions, dtype=dtype)
    if dimensions == 1:
        values = values[0]
    return polynomial_from_attributes(keys, values)
