"""Convert numpoly.ndpoly to sympy polynomial."""
from __future__ import annotations
from typing import Any

import numpy
import numpoly

from .baseclass import PolyLike


def to_sympy(poly: PolyLike) -> Any:
    """
    Convert numpoly object to sympy object, or array of sympy objects.

    Args:
        poly:
            Polynomial object to convert to sympy.

    Returns:
        If scalar, a sympy expression object, or if array, numpy.array with
        expression object values.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[1, q0**3], [q1-1, -3*q0]])
        >>> sympy_poly = to_sympy(poly)
        >>> sympy_poly
        array([[1, q0**3],
               [q1 - 1, -3*q0]], dtype=object)
        >>> type(sympy_poly.item(-1))
        <class 'sympy.core.mul.Mul'>

    """
    poly = numpoly.aspolynomial(poly)
    if poly.shape:
        return numpy.array([to_sympy(poly_) for poly_ in poly])
    from sympy import symbols  # type: ignore
    locals_ = dict(zip(poly.names, symbols(poly.names)))
    polynomial = eval(str(poly), locals_, {})  # pylint: disable=eval-used
    return polynomial
