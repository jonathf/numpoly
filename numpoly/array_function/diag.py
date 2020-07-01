"""Extract a diagonal or construct a diagonal array."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.diag)
def diag(y, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    Args:
        v (numpoly.ndpoly):
            If `v` is a 2-D array, return a copy of its `k`-th diagonal.
            If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
            diagonal.
        k (int):
            Diagonal in question. Use `k > 0` for diagonals above the main
            diagonal, and `k < 0` for diagonals below the main diagonal.

    Returns:
        (numpoly.ndpoly):
            The extracted diagonal or constructed diagonal array.

    Examples:
        >>> poly = numpoly.monomial(9).reshape(3, 3)
        >>> poly
        polynomial([[1, q0, q0**2],
                    [q0**3, q0**4, q0**5],
                    [q0**6, q0**7, q0**8]])
        >>> numpoly.diag(poly)
        polynomial([1, q0**4, q0**8])
        >>> numpoly.diag(poly, k=1)
        polynomial([q0, q0**5])
        >>> numpoly.diag(poly, k=-1)
        polynomial([q0**3, q0**7])

    """
    y = numpoly.aspolynomial(y)
    out = numpy.diag(y.values, k=k)
    return numpoly.polynomial(out, names=y.names)
