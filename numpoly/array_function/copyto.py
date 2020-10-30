"""Copy values from one array to another, broadcasting as necessary."""
import logging
import numpy
import numpoly

from ..dispatch import implements, simple_dispatch


@implements(numpy.copyto)
def copyto(dst, src, casting="same_kind", where=True):
    """
    Copy values from one array to another, broadcasting as necessary.

    Raises a TypeError if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    Args:
        dst (numpoly.ndpoly):
            The array into which values are copied.
        src (numpoly.ndpoly):
            The array from which values are copied.
        casting (str):
            Controls what kind of data casting may occur when copying.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.
        where (bool, numpy.ndarray):
            A boolean array which is broadcasted to match the dimensions
            of `dst`, and selects elements to copy from `src` to `dst`
            wherever it contains the value True.

    Examples:
        >>> q0 = numpoly.variable()
        >>> poly1 = numpoly.polynomial([1, q0**2, q0])
        >>> poly2 = numpoly.polynomial([2, q0, 3])
        >>> numpoly.copyto(poly1, poly2, where=[True, False, True])
        >>> poly1
        polynomial([2, q0**2, 3])
        >>> numpoly.copyto(poly1, poly2)
        >>> poly1
        polynomial([2, q0, 3])

    """
    logger = logging.getLogger(__name__)
    src = numpoly.aspolynomial(src)
    assert isinstance(dst, numpy.ndarray)
    if not isinstance(dst, numpoly.ndpoly):
        if dst.dtype.names is None:
            if src.isconstant():
                return numpy.copyto(
                    dst, src.tonumpy(), casting=casting, where=where)
            raise ValueError(
                "Could not convert src %s to dst %s" % (src, dst))
        if casting != "unsafe":
            raise ValueError(
                "could not safely convert src %s to dst %s" % (src, dst))
        logger.warning("Copying ndpoly input into ndarray")
        logger.warning("You might need to cast `numpoly.polynomial(dst)`.")
        logger.warning("Indeterminant names might be lost.")
        dst_keys = dst.dtype.names
    else:
        dst_keys = dst.keys
        src, _ = numpoly.align_indeterminants(src, dst)

    missing_keys = set(src.keys).difference(dst_keys)
    if missing_keys:
        raise ValueError("memory layouts are incompatible: %s" % missing_keys)

    for key in dst_keys:
        if key in src.keys:
            numpy.copyto(dst[key], src[key], casting=casting, where=where)
        else:
            numpy.copyto(dst[key], numpy.array(False, dtype=dst[key].dtype),
                         casting=casting, where=where)
