"""Align polynomials."""
import numpy

from . import construct


def align_polynomials(
        poly1,
        poly2,
        adjust_dim=True,
        adjust_shape=True,
):
    """
    Align polynomial such that dimensionality, shape, etc. are compatible.

    Args:
        poly1 (numpoly.ndpoly, array_like):
            Polynomial to adjust shape for.
        poly2 (numpoly.ndpoly, array_like):
            Polynomial to adjust shape for.
        adjust_dim (bool):
            Adjust dimensionality of polynomial
        adjust_shape (bool):
            Adjusting shape of polynomials using broadcasting.

    Returns:
        (numpoly, numpoly):
            Same as ``poly1`` and ``poly2``, but internal adjustments made to
            make them compatible for further operations.

    Examples:
        >>> x = numpoly.variable(1)
        >>> poly1 = numpoly.polynomial([[1, x], [4, x]])
        >>> poly2 = 3*numpoly.variable(2)[1]
        >>> poly1_, poly2_ = align_polynomials(poly1, poly2)
        >>> numpy.all(poly1 == poly1_)
        True
        >>> numpy.all(poly2 == poly2_)
        True
        >>> print(poly1.exponents)
        [[0]
         [1]]
        >>> print(poly1_.exponents)
        [[0 0]
         [1 0]]
        >>> print(poly2.shape)
        ()
        >>> print(poly2_.shape)
        (2, 2)
    """
    poly1 = construct.polynomial(poly1)
    poly2 = construct.polynomial(poly2)

    if adjust_dim:
        dim1 = poly1.exponents.shape[1]
        dim2 = poly2.exponents.shape[1]
        if dim1 < dim2:
            poly2, poly1 = align_polynomials(
                poly2, poly1, adjust_dim=True, adjust_shape=False)

        elif dim1 > dim2:
            exponents = numpy.hstack([
                poly2.exponents,
                numpy.zeros((len(poly2.exponents), dim1-dim2), dtype=int),
            ])
            poly2 = construct.polynomial_from_attributes(
                exponents, poly2.coefficients)
            assert dim1 == poly2.exponents.shape[1]

    if adjust_shape:
        shapedelta = len(poly2.shape)-len(poly1.shape)
        if shapedelta < 0:
            poly2, poly1 = align_polynomials(
                poly2, poly1, adjust_dim=False, adjust_shape=True)

        elif shapedelta > 0:
            coefficients = numpy.array(poly1.coefficients)[
                (slice(None),)+(numpy.newaxis,)*shapedelta]
            common = (numpy.ones(coefficients.shape[1:], dtype=bool)|
                      numpy.ones(poly2.shape, dtype=bool))
            poly1 = construct.polynomial_from_attributes(
                poly1.exponents, coefficients*common)
            poly2 = construct.polynomial_from_attributes(
                poly2.exponents, numpy.array(poly2.coefficients)*common)
        assert poly1.shape == poly2.shape

    return poly1, poly2
