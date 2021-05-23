"""Simple constructor to create single variables to create polynomials."""
from typing import Optional

import numpy.typing

import numpoly
from ..baseclass import ndpoly


def variable(
        dimensions: int = 1,
        asarray: bool = False,
        dtype: numpy.typing.DTypeLike = "i8",
        allocation: Optional[int] = None,
) -> ndpoly:
    """
    Construct variables that can be used to construct polynomials.

    Args:
        dimensions:
            Number of dimensions in the array.
        asarray:
            Enforce output as array even in the case where there is only one
            variable.
        dtype:
            The data type of the polynomial coefficients.
        allocation:
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Returns:
        Polynomial array with unit components in each dimension.

    Examples:
        >>> numpoly.variable()
        polynomial(q0)
        >>> q0, q1, q2 = numpoly.variable(3)
        >>> q1+1
        polynomial(q1+1)
        >>> numpoly.polynomial([q2**3, q1+q2, 1])
        polynomial([q2**3, q2+q1, 1])

    """
    return numpoly.symbols(
        names=f"{numpoly.get_options()['default_varname']}:{dimensions:d}",
        asarray=asarray,
        dtype=dtype,
        allocation=allocation,
    )
