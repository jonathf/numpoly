"""Gives a new shape to an array without changing its data."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.reshape)
def reshape(a, newshape, order="C"):
    """
    Give a new shape to an array without changing its data.

    Args:
        a (numpoly.ndpoly):
            Array to be reshaped.
        newshape (int, Tuple[int, ...]):
            The new shape should be compatible with the original shape. If an
            integer, then the result will be a 1-D array of that length. One
            shape dimension can be -1. In this case, the value is inferred from
            the length of the array and remaining dimensions.
        order (str):
            Read the elements of `a` using this index order, and place the
            elements into the reshaped array using this index order.  'C' means
            to read / write the elements using C-like index order, with the
            last axis index changing fastest, back to the first axis index
            changing slowest. 'F' means to read / write the elements using
            Fortran-like index order, with the first index changing fastest,
            and the last index changing slowest. Note that the 'C' and 'F'
            options take no account of the memory layout of the underlying
            array, and only refer to the order of indexing. 'A' means to read
            / write the elements in Fortran-like index order if `a` is Fortran
            *contiguous* in memory, C-like order otherwise.

    Returns:
        (ndarray):
            This will be a new view object if possible; otherwise, it will be
            a copy.  Note there is no guarantee of the *memory layout* (C- or
            Fortran- contiguous) of the returned array.

    Examples:
        >>> numpoly.reshape([1, 2, 3, 4], (2, 2))
        polynomial([[1, 2],
                    [3, 4]])
        >>> numpoly.reshape(numpoly.monomial(6), (3, 2))
        polynomial([[1, q0],
                    [q0**2, q0**3],
                    [q0**4, q0**5]])

    """
    poly = numpoly.aspolynomial(a)
    array = numpy.reshape(poly.values, newshape=newshape, order=order)
    return numpoly.aspolynomial(array, names=poly.indeterminants)
