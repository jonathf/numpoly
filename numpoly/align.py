"""Align polynomials."""
import numpy

from . import construct


def align_polynomials(
        poly1,
        poly2,
):
    """
    Align polynomial such that dimensionality, shape, etc. are compatible.

    Args:
        poly1 (numpoly.ndpoly, array_like):
            Polynomial to adjust shape for.
        poly2 (numpoly.ndpoly, array_like):
            Polynomial to adjust shape for.

    Returns:
        (numpoly, numpoly):
            Same as ``poly1`` and ``poly2``, but internal adjustments made to
            make them compatible for further operations.
    """
    poly1, poly2 = align_polynomial_shape(poly1, poly2)
    poly1, poly2 = align_polynomial_indeterminants(poly1, poly2)
    return poly1, poly2


def align_polynomial_shape(poly1, poly2):
    """
    Align polynomial by shape.

    Basically numpy broadcasting.
    """
    poly1 = construct.polynomial(poly1)
    poly2 = construct.polynomial(poly2)

    shapedelta = len(poly2.shape)-len(poly1.shape)
    if shapedelta < 0:
        poly2, poly1 = align_polynomial_shape(poly2, poly1)

    elif shapedelta > 0:
        coefficients = numpy.array(poly1.coefficients)[
            (slice(None),)+(numpy.newaxis,)*shapedelta]
        common = (numpy.ones(coefficients.shape[1:], dtype=bool)|
                    numpy.ones(poly2.shape, dtype=bool))
        poly1 = construct.polynomial_from_attributes(
            exponents=poly1.exponents,
            coefficients=coefficients*common,
            indeterminants=poly1.indeterminants,
            trim=False,
        )
        poly2 = construct.polynomial_from_attributes(
            exponents=poly2.exponents,
            coefficients=numpy.array(poly2.coefficients)*common,
            indeterminants=poly2.indeterminants,
            trim=False,
        )

    else:
        poly1 = poly1.copy()
        poly2 = poly2.copy()

    assert poly1.shape == poly2.shape

    return poly1, poly2


def align_polynomial_indeterminants(*polys):
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
