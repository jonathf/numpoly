"""Return a scalar type which is common to the input arrays."""
from __future__ import annotations
import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.common_type)
def common_type(*arrays: PolyLike) -> numpy.dtype:
    """
    Return a scalar type which is common to the input arrays.

    The return type will always be an inexact (i.e. floating point) scalar
    type, even if all the arrays are integer arrays. If one of the inputs is an
    integer array, the minimum precision type that is returned is a 64-bit
    floating point dtype.

    All input arrays except int64 and uint64 can be safely cast to the
    returned dtype without loss of information.

    Args:
        arrays:
            Input arrays.

    Return:
        Data type code.

    Examples:
        >>> scalar = numpy.array(2, dtype=numpy.float32)
        >>> numpoly.common_type(scalar) is numpy.float32
        True
        >>> q0 = numpoly.variable()
        >>> numpoly.common_type(q0) is numpy.float64
        True
        >>> numpoly.common_type(q0, 1j) is numpy.complex128
        True

    """
    arrays_ = [numpoly.aspolynomial(array) for array in arrays]
    arrays_ = [array.values[array.keys[0]] for array in arrays_]
    return numpy.common_type(*arrays_)
