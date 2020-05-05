"""Return element-wise quotient and remainder simultaneously."""
import numpy
import numpoly

from ..dispatch import implements_ufunc

DIVMOD_ERROR_MSG = """
Division-remainder involving polynomial division differs from numerical division;
Use ``numpoly.poly_divmod`` to get polynomial division-remainder."""


@implements_ufunc(numpy.divmod)
def divmod(x1, x2, out=(None, None), where=True, **kwargs):
    """
    Return element-wise quotient and remainder simultaneously.

    ``numpoly.divmod(x, y)`` is equivalent to ``(x // y, x % y)``, but faster
    because it avoids redundant work. It is used to implement the Python
    built-in function ``divmod`` on arrays.

    Args:
        x1 (numpoly.ndpoly):
            Dividend array.
        x2 (numpoly.ndpoly):
            Divisor array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a..dispatch shape (which becomes the shape of the
            output).
        out (Tuple[Optional[numpoly.ndpoly], Optional[numpoly.ndpoly]]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (Optional[numpy.ndarray]):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        (Tuple[numpoly.ndpoly, numpoly.ndpoly]):
            Element-wise quotient and remainder resulting from floor division.

    Raises:
        numpoly.baseclass.FeatureNotSupported:
            If either `x1` or `x2` contains indeterminants, numerical division
            is no longer possible and an error is raised instead. For
            polynomial division-remainder see ``numpoly.poly_divmod``.

    Examples:
        >>> numpoly.divmod([1, 22, 444], 4)
        (polynomial([0, 5, 111]), polynomial([1, 2, 0]))
        >>> numpoly.divmod([1, 22, 444], numpoly.symbols("x"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        numpoly.baseclass.FeatureNotSupported:
        Division-remainder involving polynomial division differs from numerical division;
        Use ``numpoly.poly_divmod`` to get polynomial division-remainder.

    """
    x1, x2 = numpoly.align_polynomials(x1, x2)
    if not x1.isconstant() or not x2.isconstant():
        raise numpoly.FeatureNotSupported(DIVMOD_ERROR_MSG)
    quotient, remainder = numpy.divmod(
        x1.tonumpy(), x2.tonumpy(), out=out, where=where, **kwargs)
    return numpoly.polynomial(quotient), numpoly.polynomial(remainder)
