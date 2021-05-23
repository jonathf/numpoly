"""Adjust the dimensions of a polynomial."""
from __future__ import annotations
from typing import Optional

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike


def set_dimensions(poly: PolyLike, dimensions: Optional[int] = None) -> ndpoly:
    """
    Adjust the dimensions of a polynomial.

    Args:
        poly:
            Input polynomial
        dimensions:
            The dimensions of the output polynomial. If omitted, increase
            polynomial with one dimension.

    Returns:
        Polynomials with no internal dimensions. Unless the new dim is smaller
        then `poly`'s dimensions, the polynomial should have the same content.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = q0*q1-q0**2
        >>> numpoly.set_dimensions(poly, 1)
        polynomial(-q0**2)
        >>> numpoly.set_dimensions(poly, 3)
        polynomial(q0*q1-q0**2)
        >>> numpoly.set_dimensions(poly).names
        ('q0', 'q1', 'q2')

    """
    poly = numpoly.aspolynomial(poly)
    if dimensions is None:
        dimensions = len(poly.names)+1
    diff = dimensions-len(poly.names)
    if diff > 0:
        padding = numpy.zeros((len(poly.exponents), diff), dtype="uint32")
        exponents = numpy.hstack([poly.exponents, padding])
        coefficients = poly.coefficients
        varname = numpoly.get_options()["default_varname"]
        names_ = list(poly.names)
        idx = 0
        while len(names_) < dimensions:
            if f"{varname}{idx}" not in names_:
                names_.append(f"{varname}{idx}")
            idx += 1

        indices = numpy.lexsort([names_])
        exponents = exponents[:, indices]
        names = tuple(names_[idx] for idx in indices)

    elif diff < 0:
        indices = True ^ numpy.any(poly.exponents[:, dimensions:], -1)
        exponents = poly.exponents[:, :dimensions]
        exponents = exponents[indices]
        coefficients = [
            coeff for coeff, idx in zip(poly.coefficients, indices) if idx]
        names = poly.names[:dimensions]

    else:
        return poly

    return numpoly.polynomial_from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        names=names,
        dtype=poly.dtype,
        allocation=poly.allocation,
        retain_names=True,
    )
