"""Clean up polynomial attributes."""
import numpy
import numpoly


def clean_attributes(poly):
    """Clean up polynomial attributes."""
    return numpoly.ndpoly.from_attributes(
        exponents=poly.exponents,
        coefficients=poly.coefficients,
        indeterminants=poly.names,
        dtype=poly.dtype,
        clean=True,
    )


def postprocess_attributes(exponents, coefficients, indeterminants):
    """Clean up polynomial attributes."""
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
