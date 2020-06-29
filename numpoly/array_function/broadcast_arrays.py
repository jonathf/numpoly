"""Broadcast any number of arrays against each other."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.broadcast_arrays)
def broadcast_arrays(*args, **kwargs):
    """
    Broadcast any number of arrays against each other.

    Args:
        args (numpoly.ndpoly):
            The arrays to broadcast.
        subok (bool):
            If True, then sub-classes will be passed-through, otherwise the
            returned arrays will be forced to be a base-class array (default).

    Returns:
        (List[numpoly.ndpoly]):
            These arrays are views on the original arrays.  They are typically
            not contiguous.  Furthermore, more than one element of a
            broadcasted array may refer to a single memory location. If you
            need to write to the arrays, make copies first. While you can set
            the ``writable`` flag True, writing to a single output value may
            end up changing more than one location in the output array.

    Examples:
        >>> poly1 = numpoly.monomial(3).reshape(1, 3)
        >>> poly1
        polynomial([[1, q0, q0**2]])
        >>> poly2 = numpoly.monomial(2).reshape(2, 1)
        >>> poly2
        polynomial([[1],
                    [q0]])
        >>> res1, res2 = numpoly.broadcast_arrays(poly1, poly2)
        >>> res1
        polynomial([[1, q0, q0**2],
                    [1, q0, q0**2]])
        >>> res2
        polynomial([[1, 1, 1],
                    [q0, q0, q0]])

    """
    args = [numpoly.aspolynomial(arg) for arg in args]
    results = numpy.broadcast_arrays(*[arg.values for arg in args], **kwargs)
    return [numpoly.aspolynomial(result, names=arg.indeterminants)
            for result, arg in zip(results, args)]
