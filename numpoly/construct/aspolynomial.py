"""Convert the input to an polynomial array."""
import numpy
import numpoly


def aspolynomial(
        poly_like=None,
        names=None,
        dtype=None,
):
    """
    Convert the input to an polynomial array.

    Args:
        poly_like (Any):
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        names (str, Tuple[str, ...]):
            Name of the indeterminant variables. If possible to infer from
            ``poly_like``, this argument will be ignored.
        dtype (type, numpy.dtype):
            Data type used for the polynomial coefficients.

    Returns:
        (numpoly.ndpoly):
            Array interpretation of `poly_like`. No copy is performed if the
            input is already an ndpoly with matching indeterminants names and
            dtype.

    Examples:
        >>> x = numpoly.symbols("x")
        >>> numpoly.polynomial(x) is x
        False
        >>> numpoly.aspolynomial(x) is x
        True

    """
    remain = False
    if isinstance(poly_like, numpoly.ndpoly):

        remain = (dtype is None or dtype == poly_like.dtype)
        if names is not None:
            if isinstance(names, numpoly.ndpoly):
                names = names.names
            if isinstance(names, str):
                names = [names]
            if len(names) == 1 and len(poly_like.names) > 1:
                names = ["{}{}".format(names[0], idx)
                         for idx in range(len(poly_like.indeterminants))]
            remain &= names == poly_like.names

    if remain:
        return poly_like
    return numpoly.polynomial(poly_like, names=names, dtype=dtype)
