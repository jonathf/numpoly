"""Align polynomials."""
import numpy

from . import construct


def align_polynomials(*polys):
    """
    Align polynomial such that dimensionality, shape, etc. are compatible.

    Args:
        poly1, poly2, ... (numpoly.ndpoly, array_like):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.
    """
    polys = align_polynomial_shape(*polys)
    polys = align_polynomial_indeterminants(*polys)
    return polys


def align_polynomial_shape(*polys):
    """
    Align polynomial by shape.

    Args:
        poly1, poly2, ... (numpoly.ndpoly, array_like):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly1 = 4*x
        >>> poly2 = numpoly.polynomial([[2*x+1, 3*x-y]])
        >>> print(poly1, poly2)
        4*x [[1+2*x -y+3*x]]
        >>> print(poly1.shape, poly2.shape)
        () (1, 2)
        >>> poly1, poly2 = numpoly.align_polynomial_shape(poly1, poly2)
        >>> print(poly1, poly2)
        [[4*x 4*x]] [[1+2*x -y+3*x]]
        >>> print(poly1.shape, poly2.shape)
        (1, 2) (1, 2)
    """
    polys = [construct.polynomial(poly) for poly in polys]
    common = 1
    for poly in polys:
        common = numpy.ones(poly.coefficients[0].shape, dtype=int)*common

    polys = [construct.polynomial_from_attributes(
        exponents=poly.exponents,
        coefficients=[coeff*common for coeff in poly.coefficients],
        indeterminants=poly.indeterminants,
    ) for poly in polys]
    assert numpy.all(common.shape == poly.shape for poly in polys)
    return tuple(polys)


def align_polynomial_indeterminants(*polys):
    """
    Align polynomial by indeterminants.

    Args:
        poly1, poly2, ... (numpoly.ndpoly, array_like):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly1, poly2 = numpoly.polynomial([2*x+1, 3*x-y])
        >>> print(poly1, poly2)
        1+2*x -y+3*x
        >>> print(poly1.indeterminants, poly2.indeterminants)
        [x] [x y]
        >>> poly1, poly2 = numpoly.align_polynomial_indeterminants(poly1, poly2)
        >>> print(poly1, poly2)
        1+2*x -y+3*x
        >>> print(poly1.indeterminants, poly2.indeterminants)
        [x y] [x y]
    """
    polys = [construct.polynomial(poly) for poly in polys]
    common_indeterminates = sorted({
        indeterminant
        for poly in polys
        for indeterminant in poly._indeterminants
    })
    for idx, poly in enumerate(polys):
        indices = numpy.array([
            common_indeterminates.index(indeterminant)
            for indeterminant in poly._indeterminants
        ])
        exponents = numpy.zeros(
            (len(poly._exponents), len(common_indeterminates)), dtype=int)
        exponents[:, indices] = poly.exponents
        polys[idx] = construct.polynomial_from_attributes(
            exponents=exponents,
            coefficients=poly.coefficients,
            indeterminants=common_indeterminates,
            trim=False,
        )

    return tuple(polys)
