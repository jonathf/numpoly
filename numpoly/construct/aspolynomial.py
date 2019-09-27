"""Convert the input to an polynomial array."""
import numpy
import numpoly


def aspolynomial(
        poly_like=None,
        indeterminants=None,
        dtype=None,
):
    """
    Convert the input to an polynomial array.

    Args:
        poly_like (Any):
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        indeterminants (str, Tuple[str, ...]):
            Name of the indeterminant variables. If possible to infer from
            ``poly_like``, this argument will be ignored.
        dtype (type, numpy.dtype):
            Data type used for the polynomial coefficients.

    Returns:
        (numpoly.ndpoly):
            Array interpretation of `poly_like`. No copy is performed if the
            input is already an ndpoly with matching indeterminants and dtype.

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
        if indeterminants is not None:
            if isinstance(indeterminants, numpoly.ndpoly):
                indeterminants = indeterminants.names
            if isinstance(indeterminants, str):
                indeterminants = [indeterminants]
            if len(indeterminants) == 1 and len(poly_like.names) > 1:
                indeterminants = [
                    "{}{}".format(indeterminants[0], idx)
                    for idx in range(len(poly_like.indeterminants))
                ]
            remain |= indeterminants == poly_like.names

    if remain:
        return poly_like

    # assume polynomial converted to structured array
    if isinstance(poly_like, numpy.ndarray) and poly_like.dtype.names:
        keys = numpy.asarray(poly_like.dtype.names, dtype="U")
        exponents = keys.flatten().view(numpy.uint32)-48
        exponents = exponents.reshape(len(keys), -1)
        return numpoly.ndpoly(
            exponents=exponents,
            shape=poly_like.shape,
            indeterminants=indeterminants,
            dtype=dtype,
            buffer=poly_like.data
        )

    if indeterminants is None:
        return numpoly.polynomial(poly_like, dtype=dtype)
    return numpoly.polynomial(poly_like, indeterminants=indeterminants, dtype=dtype)
