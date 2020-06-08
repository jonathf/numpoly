"""Find the largest exponents in the polynomial."""
import numpy
import numpoly


def largest_exponent(poly, graded=False, reverse=False):
    """
    Find the largest exponents in the polynomial.

    As polynomials are not inherently sortable, values are sorted using the
    highest `lexicographical` ordering. Between the values that have the same
    highest ordering, the elements are sorted using the coefficients. This also
    ensures that the method behaves as expected with ``numpy.ndarray``.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to locate exponents on.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``x**2*y**2*z**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``x**3*y*z``, ``x*y**3*z``
            and ``x*y*z**3``.
        reverse (bool):
            Reverses lexicographical sorting meaning that ``x*y**3`` is
            considered bigger than ``x**3*y``, instead of the opposite.

    Returns:
        (numpy.ndarray):
            Integer array with the largest exponents in the polynomials. The
            shape is ``poly.shape + (len(poly.names),)``. The extra dimension
            is used to indicate the exponent for the different indeterminants.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> numpoly.largest_exponent([1, x+1, x**2+x+1]).T
        array([[0, 1, 2]])
        >>> numpoly.largest_exponent([1, x, y, x*y, x**3-1]).T
        array([[0, 1, 0, 1, 3],
               [0, 0, 1, 1, 0]])

    """
    poly = numpoly.aspolynomial(poly)
    shape = poly.shape
    poly = poly.ravel()
    out = numpy.zeros(poly.shape+(len(poly.names),), dtype=int)
    for idx in numpoly.glexsort(poly.exponents.T, graded=graded, reverse=reverse):
        out[poly.coefficients[idx] != 0] = poly.exponents[idx]
    return out.reshape(shape+(len(poly.names),))
