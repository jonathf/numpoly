"""Polynomials differentiation functions."""
from __future__ import annotations
from typing import Union

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike


def derivative(poly: PolyLike, *diffvars: Union[ndpoly, str, int]) -> ndpoly:
    """
    Polynomial differential operator.

    Args:
        poly:
            Polynomial to differentiate.
        diffvars:
            Singleton variables to take derivative off.

    Returns:
        Same as ``poly`` but differentiated with respect to ``diffvars``.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([1, q0, q0*q1**2+1])
        >>> poly
        polynomial([1, q0, q0*q1**2+1])
        >>> numpoly.derivative(poly, "q0")
        polynomial([0, 1, q1**2])
        >>> numpoly.derivative(poly, 0, 1)
        polynomial([0, 0, 2*q1])
        >>> numpoly.derivative(poly, q0, q0, q0)
        polynomial([0, 0, 0])

    """
    poly = poly_ref = numpoly.aspolynomial(poly)

    for diffvar in diffvars:
        if isinstance(diffvar, str):
            idx = poly.names.index(diffvar)
        elif isinstance(diffvar, int):
            idx = diffvar
        else:
            diffvar = numpoly.aspolynomial(diffvar)
            exponents, names = numpoly.remove_redundant_names(
                diffvar.exponents, diffvar.names)
            assert names is not None and len(names) == 1, "one at the time"
            assert numpy.all(exponents == 1), (
                "derivative variable assumes singletons")
            idx = poly.names.index(names[0])

        exponents = poly.exponents
        coefficients = [
            (exponent[idx]*coefficient.T).T
            for exponent, coefficient in zip(exponents, poly.coefficients)
        ]
        exponents[:, idx] -= 1
        assert not numpy.any(exponents < 0)

        poly = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=poly_ref.names,
        )
        poly, poly_ref = numpoly.align_polynomials(poly, poly_ref)
    return poly


def gradient(poly: PolyLike) -> ndpoly:
    """
    Polynomial gradient operator.

    Args:
        poly:
            Polynomial to differentiate. If polynomial vector is provided,
            the Jacobi-matrix is returned instead.

    Returns:
        Same as ``poly`` but with an extra first dimensions, one for each
        variable in ``poly.indeterminants``, filled with gradient values.

    Examples:
        >>> q0 = numpoly.variable()
        >>> poly = 5*q0**5+4
        >>> numpoly.gradient(poly)
        polynomial([25*q0**4])
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = 4*q0**3+2*q1**2+3
        >>> numpoly.gradient(poly)
        polynomial([12*q0**2, 4*q1])
        >>> poly = numpoly.polynomial([1, q0**3, q0*q1**2+1])
        >>> numpoly.gradient(poly)
        polynomial([[0, 3*q0**2, q1**2],
                    [0, 0, 2*q0*q1]])

    """
    poly = numpoly.aspolynomial(poly)
    polys = [derivative(poly, diffvar)[numpy.newaxis]
             for diffvar in poly.names]
    return numpoly.concatenate(polys, axis=0)


def hessian(poly: PolyLike) -> ndpoly:
    """
    Construct Hessian matrix of polynomials.

    Make Hessian matrix out of a polynomial. Tensor is returned if polynomial
    is a vector.

    Args:
        poly:
            Polynomial to differentiate.

    Returns:
        Same as ``poly`` but with two extra dimensions, one for each
        variable in ``poly.indeterminants``, filled with Hessian values.

    Examples:
        >>> q0 = numpoly.variable()
        >>> poly = 5*q0**5+4
        >>> numpoly.hessian(poly)
        polynomial([[100*q0**3]])
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = 4*q0**3+2*q1**2+3
        >>> numpoly.hessian(poly)
        polynomial([[24*q0, 0],
                    [0, 4]])
        >>> poly = numpoly.polynomial([1, q0, q0*q1**2+1])
        >>> numpoly.hessian(poly)
        polynomial([[[0, 0, 0],
                     [0, 0, 2*q1]],
        <BLANKLINE>
                    [[0, 0, 2*q1],
                     [0, 0, 2*q0]]])

    """
    return gradient(gradient(poly))
