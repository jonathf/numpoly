"""Find a polynomial with the given sequence of roots."""

from typing import Optional, Sequence

import numpy
import numpy.typing

import numpoly
from ..baseclass import ndpoly


def polynomial_from_roots(
    seq_of_zeros: Sequence[int],
    dtype: Optional[numpy.typing.DTypeLike] = None,
) -> ndpoly:
    """
    Find the coefficients of a polynomial with the given sequence of roots.

    Returns the coefficients of the polynomial whose leading coefficient is one
    for the given sequence of zeros (multiple roots must be included in the
    sequence as many times as their multiplicity; see Examples). A square
    matrix (or array, which will be treated as a matrix) can also be given, in
    which case the coefficients of the characteristic polynomial of the matrix
    are returned.

    Args:
        seq_of_zeros:
            A sequence of polynomial roots, or a square array or matrix object.
            Either shape (N,) or (N, N).
        dtype:
            Any object that can be interpreted as a numpy data type.

    Return:
        1-D polynomial which have `seq_of_zeros` as roots.
        Leading coefficient is always 1.

    Raise:
        ValueError:
            If input is the wrong shape (the input must be a 1-D or square
            2-D array).

    Example:
        >>> numpoly.polynomial_from_roots((0, 0, 0))
        polynomial(q0**3)
        >>> numpoly.polynomial_from_roots((-0.5, 0, 0.5))
        polynomial(q0**3-0.25*q0)

    """
    exponent = numpy.arange(len(seq_of_zeros), -1, -1, dtype=int)
    basis = numpoly.variable(dtype=dtype) ** exponent
    return numpoly.sum(numpy.poly(seq_of_zeros) * basis)
