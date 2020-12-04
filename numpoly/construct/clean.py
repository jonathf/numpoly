"""Clean up polynomial attributes."""
from six import string_types
import numpy
import numpoly


class PolynomialConstructionError(ValueError):
    """Error related to construction of polynomial."""


def clean_attributes(
        poly,
        retain_coefficients=None,
        retain_names=None,
):
    """
    Clean up polynomial attributes.

    Some operations results in polynomial with structures that are redundant.
    This includes extra unused indeterminants, and extra terms consisting of
    only zeros.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to clean up.
        retain_coefficients (Optional[bool]):
            Do not remove redundant coefficients. If omitted use global
            defaults.
        retain_names (Optional[bool]):
            Do not remove redundant names. If omitted use global defaults.

    Returns:
        Same as `poly`, but with attributes cleaned up.

    Examples:
        >>> q0, _ = numpoly.align_polynomials(*numpoly.variable(2))
        >>> q0.indeterminants
        polynomial([q0, q1])
        >>> q0.exponents
        array([[1, 0],
               [0, 1]], dtype=uint32)
        >>> q0 = numpoly.clean_attributes(
        ...     q0, retain_coefficients=False, retain_names=False)
        >>> q0.indeterminants
        polynomial([q0])
        >>> q0.exponents
        array([[1]], dtype=uint32)

    """
    return numpoly.ndpoly.from_attributes(
        exponents=poly.exponents,
        coefficients=poly.coefficients,
        names=poly.names,
        dtype=poly.dtype,
        retain_coefficients=retain_coefficients,
        retain_names=retain_names,
    )


def postprocess_attributes(
        exponents,
        coefficients,
        names=None,
        retain_coefficients=None,
        retain_names=None,
):
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
        names (Union[None, Sequence[str], numpoly.ndpoly]):
            The indeterminant names, either as string names or as
            simple polynomials. Must correspond to the exponents by having
            length ``D``.
        retain_coefficients (Optional[bool]):
            Do not remove redundant coefficients. If omitted use global
            defaults.
        retain_names (Optional[bool]):
            Do not remove redundant names. If omitted use global defaults.

    Returns:
        (numpoly.ndarray, List[numpy.ndarray], Optional[Tuple[str, ...]]):
            Same as input, but post-processed.

    """
    exponents = numpy.asarray(exponents)
    if exponents.ndim != 2:
        raise PolynomialConstructionError(
            "expected exponents.ndim == 2; found %d" % exponents.ndim)

    coefficients = [numpy.asarray(coefficient) for coefficient in coefficients]
    if coefficients and len(exponents) != len(coefficients):
        raise PolynomialConstructionError(
            "expected len(exponents) == len(coefficients); found %d != %d" % (
                len(exponents), len(coefficients)))

    if retain_coefficients is None:
        retain_coefficients = numpoly.get_options()["retain_coefficients"]
    if not retain_coefficients and coefficients:
        exponents, coefficients = remove_redundant_coefficients(exponents, coefficients)

    if isinstance(names, numpoly.ndpoly):
        names = names.names
    if isinstance(names, string_types):
        if exponents.shape[1] > 1:
            names = ["%s%d" % (names, idx)
                     for idx in range(exponents.shape[1])]
        else:
            names = [names]
    if names:
        if len(names) != exponents.shape[1]:
            raise PolynomialConstructionError(
                "Name length incompatible exponent length; "
                "len%s != %d" % (names, exponents.shape[1]))
        if sorted(set(names)) != sorted(names):
            raise PolynomialConstructionError(
                "Duplicate indeterminant names: %s" % names)

    if retain_names is None:
        retain_names = numpoly.get_options()["retain_names"]
    if not retain_names:
        exponents, names = remove_redundant_names(exponents, names)

    exponents_, count = numpy.unique(exponents, return_counts=True, axis=0)
    if numpy.any(count > 1):
        raise PolynomialConstructionError(
            "Duplicate exponent keys found: %s" % exponents_[count > 1][0])

    return exponents, coefficients, names


def remove_redundant_coefficients(exponents, coefficients):
    """
    Remove coefficients if they are redundant to the polynomial representation.

    Will always keep at least one coefficient.

    Args:
        exponents (numpy.ndarray):
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is
            the number of dimensions.
        coefficients (Sequence[numpy.ndarray]):
            The polynomial coefficients. Must correspond to `exponents` by
            having the same length ``N``.

    Returns:
        Same as inputs, but with redundant exponent rows and coefficients
        removed. This corresponds to ``N`` being lowered.

    Examples:
        >>> exponents, coefficients = remove_redundant_coefficients(
        ...     [[0, 0], [0, 1]], [1, 2])
        >>> exponents
        array([[0, 0],
               [0, 1]])
        >>> coefficients
        [array(1), array(2)]
        >>> remove_redundant_coefficients(
        ...     [[0, 0], [0, 1]], [1, 0])
        (array([[0, 0]]), [array(1)])
        >>> remove_redundant_coefficients([[0]], [])
        (array([[0]]), [])

    """
    exponents = numpy.asarray(exponents)
    coefficients = [numpy.asarray(coefficient) for coefficient in coefficients]

    if not coefficients:
        assert exponents.shape == (1, 1)

    else:
        elements = list(zip(*[
            (exponent, coefficient)
            for exponent, coefficient in zip(exponents, coefficients)
            if numpy.any(coefficient) or not numpy.any(exponent)
        ]))
        exponents = numpy.asarray(elements[0], dtype=int)
        coefficients = list(elements[1])
    return exponents, coefficients


def remove_redundant_names(exponents, names):
    """
    Remove names if they are redundant to the polynomial representation.

    Will always keep at least one dimension.

    Args:
        exponents (numpy.ndarray):
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is
            the number of dimensions.
        names (Sequence[str]):
            The indeterminant names, either as string names or as
            simple polynomials. Must correspond to the exponents by having
            length ``D``.

    Returns:
        Same as inputs, but with redundant exponent columns and names removed.
        This corresponds to ``D`` being lowered.

    Examples:
        >>> exponents, names = remove_redundant_names(
        ...     [[0, 0], [1, 1]], ["q0", "q1"])
        >>> exponents
        array([[0, 0],
               [1, 1]])
        >>> names
        ['q0', 'q1']
        >>> exponents, names = remove_redundant_names(
        ...     [[0, 0], [0, 1]], ["q0", "q1"])
        >>> exponents
        array([[0],
               [1]])
        >>> names
        ['q1']
        >>> exponents, names = remove_redundant_names(
        ...     [[0, 0]], ["q0", "q1"])
        >>> exponents
        array([[0]])
        >>> names
        ['q0']

    """
    exponents = numpy.asarray(exponents)

    indices = numpy.any(exponents != 0, 0)
    if not numpy.any(indices):
        indices[0] = True

    if names:
        names = list(names)
        names = numpy.array(names)[indices].tolist()
    exponents = exponents[:, indices]
    return exponents, names
