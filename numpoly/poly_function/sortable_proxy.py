"""Create a numerical proxy for a polynomial to allow compare."""
import numpy
import numpoly


def sortable_proxy(poly, ordering="G"):
    """
    Create a numerical proxy for a polynomial to allow compare.

    As polynomials are not inherently sortable, values are sorted using the
    highest `lexicographical` ordering. Between the values that have the same
    highest ordering, the elements are sorted using the coefficients. This also
    ensures that the method behaves as expected with ``numpy.ndarray``.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to convert into something sortable.
        ordering (str):
            Short hand for the criteria to sort the indices by.

            ``G``
                Graded sorting, meaning the indices are always sorted by the
                index sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will
                therefore be consider larger than both ``(3, 1, 1)`` and
                ``(1, 1, 3)``.
            ``R``
                Reversed, meaning the biggest values are in the front instead
                of the back.
            ``I``
                Inverse lexicographical sorting meaning that ``(1, 3)`` is
                considered bigger than ``(3, 1)``, instead of the opposite.

    Returns:
        (numpy.ndarray):
            Integer array where ``A > B`` is retained for the giving rule of
            ``ordering``.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([x**2, 2*x, 3*y, 4*x, 5])
        >>> numpoly.sortable_proxy(poly)
        array([4, 2, 1, 3, 0])
        >>> numpoly.sortable_proxy(poly, ordering="GR")
        array([4, 1, 3, 2, 0])
        >>> numpoly.sortable_proxy([8, 4, 10, -100])
        array([2, 1, 3, 0])
        >>> numpoly.sortable_proxy([[8, 4], [10, -100]])
        array([[2, 1],
               [3, 0]])

    """
    poly = numpoly.aspolynomial(poly)
    coefficients = poly.coefficients
    proxy = numpy.tile(-1, poly.shape)

    for idx in numpoly.bsort(poly.exponents.T, ordering=ordering):

        indices = coefficients[idx] != 0
        values = numpy.argsort(coefficients[idx][indices])
        proxy[indices] = numpy.argsort(values)+numpy.max(proxy)+1

    proxy = numpy.argsort(numpy.argsort(proxy.ravel())).reshape(proxy.shape)
    return proxy
