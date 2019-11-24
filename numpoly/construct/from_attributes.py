"""Construct polynomial from polynomial attributes."""
import numpoly

from . import clean as clean_


def polynomial_from_attributes(
        exponents,
        coefficients,
        names,
        dtype=None,
        clean=True,
):
    """
    Construct polynomial from polynomial attributes.

    Args:
        exponents (numpy.ndarray):
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is
            the number of dimensions.
        coefficients (Iterable[numpy.ndarray]):
            The polynomial coefficients. Must correspond to `exponents` by
            having the same length ``N``.
        names (Union[Sequence[str], numpoly.ndpoly]):
            The indeterminant names, either as string names or as
            simple polynomials. Must correspond to the exponents by having
            length ``D``.
        dtype (Optional[numpy.dtype]):
            The data type of the polynomial. If omitted, extract from
            `coefficients`.
        clean (bool):
            Clean up attributes, removing redundant indeterminant names and
            exponents.

    Returns:
        (numpoly.ndpoly):
            Polynomial array with attributes determined by the input.

    Examples:
        >>> numpoly.ndpoly.from_attributes(
        ...     exponents=[(0,), (1,)],
        ...     coefficients=[[1, 0], [0, 1]],
        ...     names=("x",),
        ... )
        polynomial([1, x])
        >>> numpoly.ndpoly.from_attributes(
        ...     exponents=[(0, 0, 0), (1, 1, 2)],
        ...     coefficients=[4, -1],
        ...     names=("x", "y", "z"),
        ... )
        polynomial(4-x*y*z**2)
        >>> numpoly.ndpoly.from_attributes(
        ...     exponents=[(0,)],
        ...     coefficients=[0],
        ... )
        polynomial(0)

    """
    if clean:
        exponents, coefficients, names = clean_.postprocess_attributes(
            exponents, coefficients, names)
    dtype = coefficients[0].dtype if dtype is None else dtype
    poly = numpoly.ndpoly(
        exponents=exponents,
        shape=coefficients[0].shape,
        names=names,
        dtype=dtype,
    )
    for exponent, values in zip(poly.keys, coefficients):
        poly[exponent] = values
    return poly
