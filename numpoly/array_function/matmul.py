"""Matrix product of two arrays."""
import numpy
import numpoly

from ..dispatch import implements

ERROR_MESSAGE = """\
matmul: Input operand %d does not have enough dimensions \
(has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)
"""


@implements(numpy.matmul)
def matmul(x1, x2, out=None, **kwargs):
    """
    Matrix product of two arrays.

    Args:
        x1, x2 (numpoly.ndpoly):
            Input arrays, scalars not allowed.
        out (Optional[numpoly.ndpoly]):
            A location into which the result is stored. If provided, it must have
            a shape that matches the signature `(n,k),(k,m)->(n,m)`. If not
            provided or `None`, a freshly-allocated array is returned.
            n,k,m

    Returns:
        (numpoly.ndpoly):
            The matrix product of the inputs. This is a scalar only when both
            x1, x2 are 1-d vectors.

    Raises:
        ValueError:
            If the last dimension of `x1` is not the same size as
            the second-to-last dimension of `x2`.

    Examples:
        >>> poly = numpoly.variable(4).reshape(2, 2)
        >>> poly
        polynomial([[q0, q1],
                    [q2, q3]])
        >>> numpoly.matmul(poly, [[0, 1], [2, 3]])
        polynomial([[2*q1, 3*q1+q0],
                    [2*q3, 3*q3+q2]])
        >>> numpoly.matmul(poly, [4, 5])
        polynomial([[4*q1+4*q0, 5*q1+5*q0],
                    [4*q3+4*q2, 5*q3+5*q2]])
        >>> numpoly.matmul(*poly)
        polynomial([q1*q2+q0*q2, q1*q3+q0*q3])

    """
    x1 = numpoly.aspolynomial(x1)
    x2 = numpoly.aspolynomial(x2)
    if not x1.shape:
        raise ValueError(ERROR_MESSAGE % 0)
    if not x2.shape:
        raise ValueError(ERROR_MESSAGE % 1)
    x1 = numpoly.reshape(x1, x1.shape+(1,))
    x2 = numpoly.reshape(x2, x2.shape[:-2]+(1,)+x2.shape[-2:])
    x1, x2 = numpoly.broadcast_arrays(x1, x2)
    out = numpoly.multiply(x1, x2, out=out, **kwargs)
    out = numpoly.sum(out, axis=-2)
    return out
