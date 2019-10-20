"""Clean up polynomial attributes."""
import numpy
import numpoly


class PolynomialConstructionError(ValueError):
    """Error related to construction of polynomial."""


def clean_attributes(poly):
    """
    Clean up polynomial attributes.

    Some operations results in polynomial with structures that are redundant.
    This includes extra unused indeterminants, and extra terms consisting of
    only zeros.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to clean up.

    Returns:
        Same as `poly`, but with attributes cleaned up.

    Examples:
        >>> x, _ = numpoly.align_polynomials(*numpoly.symbols("x y"))
        >>> x.indeterminants
        polynomial([x, y])
        >>> x.exponents
        array([[1, 0],
               [0, 1]], dtype=uint32)
        >>> x = numpoly.clean_attributes(x)
        >>> x.indeterminants
        polynomial([x])
        >>> x.exponents
        array([[1]], dtype=uint32)

    """
    return numpoly.ndpoly.from_attributes(
        exponents=poly.exponents,
        coefficients=poly.coefficients,
        indeterminants=poly.names,
        dtype=poly.dtype,
        clean=True,
    )


def postprocess_attributes(exponents, coefficients, indeterminants=None):
    """
    Clean up polynomial attributes.

    Args:
        exponents (numpy.ndarray):
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is
            the number of dimensions.
        coefficients (Sequence[numpy.ndarray]):
            The polynomial coefficients. Must correspond to `exponents` by
            having the same length ``N``.
        indeterminants (Union[None, Sequence[str], numpoly.ndpoly]):
            The indeterminants variables, either as string names or as
            simple polynomials. Must correspond to the exponents by having
            length ``D``.

    Returns:
        (numpoly.ndarray, List[numpy.ndarray], Optional[Tuple[str, ...]]):
            Same as input, but post-processed.

    """
    exponents = numpy.asarray(exponents)
    coefficients = [numpy.asarray(coefficient) for coefficient in coefficients]

    _validate_input(exponents, coefficients)

    exponents, coefficients = list(zip(*[
        (exponent, coefficient)
        for exponent, coefficient in zip(exponents, coefficients)
        if numpy.any(coefficient) or not numpy.any(exponent)
    ]))
    exponents = numpy.asarray(exponents, dtype=int)

    if isinstance(indeterminants, numpoly.ndpoly):
        indeterminants = indeterminants.names
    if isinstance(indeterminants, str):
        if exponents.shape[1] > 1:
            indeterminants = ["%s%d" % (indeterminants, idx)
                              for idx in range(exponents.shape[1])]
        else:
            indeterminants = [indeterminants]

    indices = numpy.any(exponents != 0, 0)
    if not numpy.any(indices):
        indices[0] = True

    if indeterminants is not None:
        if len(indeterminants) != exponents.shape[1]:
            raise PolynomialConstructionError(
                "Indeterminants length incompatible exponent length; "
                "len%s != %d" % (indeterminants, exponents.shape[1]))
        indeterminants = numpy.array(indeterminants)[indices].tolist()
        if sorted(set(indeterminants)) != sorted(indeterminants):
            raise PolynomialConstructionError(
                "Duplicate indeterminant names: %s" % indeterminants)

    exponents = exponents[:, indices]

    exponents_, count = numpy.unique(exponents, return_counts=True, axis=0)
    if numpy.any(count > 1):
        raise PolynomialConstructionError(
            "Duplicate exponent keys found: %s" % exponents_[count > 1][0])

    return exponents, coefficients, indeterminants


def _validate_input(exponents, coefficients):
    """Make sure the shape of the input is valid."""
    if exponents.ndim != 2:
        raise PolynomialConstructionError(
            "expected exponents.ndim == 2; found %d" % exponents.ndim)
    if len(exponents) != len(coefficients):
        raise PolynomialConstructionError(
            "expected len(exponents) == len(coefficients); found %d != %d" % (
                len(exponents), len(coefficients)))
