"""Return specified diagonals."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.diagonal)
def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.  If
    `a` has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal is
    returned.  The shape of the resulting array can be determined by
    removing `axis1` and `axis2` and appending an index to the right equal
    to the size of the resulting diagonals.

    Args:
        a (numpoly.ndpoly):
            Array from which the diagonals are taken.
        offset (int):
            Offset of the diagonal from the main diagonal. Can be positive or
            negative. Defaults to main diagonal (0).
        axis1 (int):
            Axis to be used as the first axis of the 2-D sub-arrays from which
            the diagonals should be taken.  Defaults to first axis (0).
        axis2 (int):
            Axis to be used as the second axis of the 2-D sub-arrays from
            which the diagonals should be taken. Defaults to second axis (1).

    Returns:
        (numpoly.ndpoly):
            If `a` is 2-D, then a 1-D array containing the diagonal and of the
            same type as `a` is returned unless `a` is a `matrix`, in which
            case a 1-D array rather than a (2-D) `matrix` is returned in order
            to maintain backward compatibility.

            If ``a.ndim > 2``, then the dimensions specified by `axis1` and
            `axis2` are removed, and a new axis inserted at the end
            corresponding to the diagonal.

    Raises:
        ValueError:
            If the dimension of `a` is less than 2.

    Examples:
        >>> poly = numpoly.monomial(9).reshape(3, 3)
        >>> poly
        polynomial([[1, q0, q0**2],
                    [q0**3, q0**4, q0**5],
                    [q0**6, q0**7, q0**8]])
        >>> numpoly.diagonal(poly)
        polynomial([1, q0**4, q0**8])

    """
    a = numpoly.aspolynomial(a)
    out = numpy.diagonal(a.values, offset=offset, axis1=axis1, axis2=axis2)
    return numpoly.polynomial(out, names=a.names)
