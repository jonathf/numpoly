"""Create a numerical proxy for a polynomial to allow compare."""
import numpy
import numpoly


def sortable_proxy(poly, graded=False, reverse=False):
    """
    Create a numerical proxy for a polynomial to allow compare.

    As polynomials are not inherently sortable, values are sorted using the
    highest `lexicographical` ordering. Between the values that have the same
    highest ordering, the elements are sorted using the coefficients. This also
    ensures that the method behaves as expected with ``numpy.ndarray``.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to convert into something sortable.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``x**2*y**2*z**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``x**3*y*z``, ``x*y**3*z``
            and ``x*y*z**3``.
        reverse (bool):
            Reverses lexicographical sorting meaning that ``x*y**3`` is
            considered smaller than ``x**3*y``, instead of the opposite.

    Returns:
        (numpy.ndarray):
            Integer array where ``a > b`` is retained for the giving rule of
            ``ordering``.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([x**2, 2*x, 3*y, 4*x, 5])
        >>> numpoly.sortable_proxy(poly)
        array([3, 1, 4, 2, 0])
        >>> numpoly.sortable_proxy(poly, reverse=True)
        array([4, 2, 1, 3, 0])
        >>> numpoly.sortable_proxy([8, 4, 10, -100])
        array([2, 1, 3, 0])
        >>> numpoly.sortable_proxy([[8, 4], [10, -100]])
        array([[2, 1],
               [3, 0]])

    """
    poly = numpoly.aspolynomial(poly)
    coefficients = poly.coefficients
    proxy = numpy.tile(-1, poly.shape)
    largest = numpoly.largest_exponent(poly, graded=graded, reverse=reverse)

    for idx in numpoly.glexsort(
            poly.exponents.T, graded=graded, reverse=reverse):

        indices = numpy.all(largest == poly.exponents[idx], axis=-1)
        values = numpy.argsort(coefficients[idx][indices])
        proxy[indices] = numpy.argsort(values)+numpy.max(proxy)+1

    proxy = numpy.argsort(numpy.argsort(proxy.ravel())).reshape(proxy.shape)
    return proxy
