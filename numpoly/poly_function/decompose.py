"""Decompose a polynomial to component form."""
from __future__ import annotations
import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike


def decompose(poly: PolyLike) -> ndpoly:
    """
    Decompose a polynomial to component form.

    In array missing values are padded with 0 to make decomposition compatible
    with ``chaospy.sum(output, 0)``.

    Args:
        poly:
            Polynomial to decompose.

    Returns:
        Decomposed polynomial with ``poly.shape==(M,)+output.shape``,
        where ``M`` is the number of components in `poly`.

    Examples:
        >>> q0 = numpoly.variable()
        >>> poly = numpoly.polynomial([q0**2-1, 2])
        >>> poly
        polynomial([q0**2-1, 2])
        >>> numpoly.decompose(poly)
        polynomial([[-1, 2],
                    [q0**2, 0]])
        >>> numpoly.sum(numpoly.decompose(poly), 0)
        polynomial([q0**2-1, 2])

    """
    poly = numpoly.aspolynomial(poly)
    return numpoly.concatenate([
        numpoly.construct.polynomial_from_attributes(
            exponents=[expon],
            coefficients=[numpy.asarray(poly.values[key])],
            names=poly.indeterminants,
            retain_coefficients=True,
            retain_names=True,
        )[numpy.newaxis] for key, expon in zip(poly.keys, poly.exponents)
    ])
