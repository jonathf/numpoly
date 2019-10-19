"""Return a scalar type which is common to the input arrays."""
import numpy
import numpoly

from .common import implements


@implements(numpy.common_type)
def common_type(*arrays):
    """
    Return a scalar type which is common to the input arrays.

    The return type will always be an inexact (i.e. floating point) scalar
    type, even if all the arrays are integer arrays. If one of the inputs is an
    integer array, the minimum precision type that is returned is a 64-bit
    floating point dtype.

    All input arrays except int64 and uint64 can be safely cast to the
    returned dtype without loss of information.

    Args:
        arrays (numpoly.ndpoly):
            Input arrays.

    Return:
        out (numpy.generic):
            Data type code.

    Examples:
        >>> numpoly.common_type(
        ...     numpy.array(2, dtype=numpy.float32)).__name__
        'float32'
        >>> numpoly.common_type(
        ...     numpoly.symbols("x")).__name__
        'float64'
        >>> numpoly.common_type(
        ...     numpy.arange(3), 1j*numpoly.symbols("x"), 45).__name__
        'complex128'

    """
    arrays = [numpoly.aspolynomial(array) for array in arrays]
    arrays = [array[array.keys[0]] for array in arrays]
    return numpy.common_type(*arrays)
