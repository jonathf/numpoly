"""Find the lead exponents for each polynomial."""
import numpy
import numpoly


def lead_exponent(poly, graded=False, reverse=False):
    """
    Find the lead exponents for each polynomial.

    As polynomials are not inherently sortable, values are sorted using the
    highest `lexicographical` ordering. Between the values that have the same
    highest ordering, the elements are sorted using the coefficients.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to locate exponents on.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**3*q1*q2``,
            ``q0*q1**3*q2`` and ``q0*q1*z**3``.
        reverse (bool):
            Reverses lexicographical sorting meaning that ``q0*q1**3`` is
            considered bigger than ``q0**3*q1``, instead of the opposite.

    Returns:
        (numpy.ndarray):
            Integer array with the largest exponents in the polynomials. The
            shape is ``poly.shape + (len(poly.names),)``. The extra dimension
            is used to indicate the exponent for the different indeterminants.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.lead_exponent([1, q0+1, q0**2+q0+1]).T
        array([[0, 1, 2]])
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.lead_exponent(
        ...     [1, q0, q1, q0*q1, q0**3-1]).T
        array([[0, 1, 0, 1, 3],
               [0, 0, 1, 1, 0]])

    """
    poly = numpoly.aspolynomial(poly)
    shape = poly.shape
    poly = poly.ravel()
    out = numpy.zeros(poly.shape+(len(poly.names),), dtype=int)
    if not poly.size:
        return out
    for idx in numpoly.glexsort(poly.exponents.T, graded=graded, reverse=reverse):
        out[poly.coefficients[idx] != 0] = poly.exponents[idx]
    return out.reshape(shape+(len(poly.names),))
