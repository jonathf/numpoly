"""Clean up polynomial attributes."""
import numpy
import numpoly


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
        array([[0, 1],
               [1, 0]], dtype=uint32)
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


def postprocess_attributes(exponents, coefficients, indeterminants):
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
        indeterminants (Union[Sequence[str], numpoly.ndpoly]):
            The indeterminants variables, either as string names or as
            simple polynomials. Must correspond to the exponents by having
            length ``D``.

    Returns:
        (numpoly.ndpoly):
            Polynomial array with attributes determined by the input.

    """
    coefficients = [numpy.asarray(coefficient) for coefficient in coefficients]
    pairs = [
        (exponent, coefficient)
        for exponent, coefficient in zip(exponents, coefficients)
        if numpy.any(coefficient) or not any(exponent)
    ]
    if pairs:
        exponents, coefficients = zip(*pairs)
    else:
        exponents = [(0,)*len(indeterminants)]
        coefficients = numpy.zeros(
            (1,)+coefficients[0].shape, dtype=coefficients[0].dtype)

    exponents = numpy.asarray(exponents, dtype=int)
    if isinstance(indeterminants, numpoly.ndpoly):
        indeterminants = indeterminants.names

    if isinstance(indeterminants, str):
        if exponents.shape[1] > 1:
            indeterminants = ["%s%d" % (indeterminants, idx)
                              for idx in range(exponents.shape[1])]
        else:
            indeterminants = [indeterminants]
    assert len(indeterminants) == exponents.shape[1], (indeterminants, exponents)

    indices = numpy.any(exponents != 0, 0)
    assert exponents.shape[1] == len(indices), (exponents, indices)
    if not numpy.any(indices):
        indices[0] = True
    exponents = exponents[:, indices]
    assert exponents.size, (exponents, indices)
    indeterminants = numpy.array(indeterminants)[indices].tolist()

    assert len(exponents.shape) == 2, exponents
    assert len(exponents) == len(coefficients)
    assert len(numpy.unique(exponents, axis=0)) == exponents.shape[0], exponents
    assert sorted(set(indeterminants)) == sorted(indeterminants)

    return exponents, coefficients, indeterminants
