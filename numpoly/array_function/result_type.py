"""Returns the type that results from applying type promotion rules."""
from __future__ import annotations
from typing import Union

import numpy
import numpy.typing
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.result_type)
def result_type(
    *arrays_and_dtypes: Union[PolyLike, numpy.typing.DTypeLike],
) -> numpy.dtype:
    """
    Return the type that results from applying type promotion rules.

    Type promotion in NumPy works similarly to the rules in languages like C++,
    with some slight differences.  When both scalars and arrays are used, the
    array's type takes precedence and the actual value of the scalar is taken
    into account.

    For example, calculating 3*a, where a is an array of 32-bit floats,
    intuitively should result in a 32-bit float output.  If the 3 is a 32-bit
    integer, the NumPy rules indicate it can't convert losslessly into a 32-bit
    float, so a 64-bit float should be the result type. By examining the value
    of the constant, '3', we see that it fits in an 8-bit integer, which can be
    cast losslessly into the 32-bit float.

    Args:
        arrays_and_dtypes:
            The operands of some operation whose result type is needed.

    Returns:
        The result type.

    Examples:
        >>> q0 = numpoly.variable(dtype="i1")
        >>> numpoly.result_type(3, numpy.arange(7, dtype="i1"), q0)
        dtype('int8')
        >>> numpoly.result_type('i4', 'c8')
        dtype('complex128')
        >>> numpoly.result_type(3.0, -2)
        dtype('float64')

    """
    values = list(arrays_and_dtypes)
    for idx, value in enumerate(values):
        if isinstance(value, numpoly.ndpoly):
            values[idx] = value.dtype
    return numpy.result_type(*values)
