"""Clean up polynomial attributes."""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import numpy
import numpy.typing

import numpoly
from ..baseclass import ndpoly


class PolynomialConstructionError(ValueError):
    """Error related to construction of polynomial."""


def clean_attributes(
        poly: ndpoly,
        retain_coefficients: Optional[bool] = None,
        retain_names: Optional[bool] = None,
) -> ndpoly:
    """
    Clean up polynomial attributes that does not affect representation.

    Some operations results in polynomial with structures that are redundant.
    This includes extra unused indeterminants and coefficients consisting of
    only zeros.

    Args:
        poly:
            Polynomial to clean up.
        retain_coefficients:
            Do not remove redundant coefficients. If omitted use global
            defaults.
        retain_names:
            Do not remove redundant names. If omitted use global defaults.

    Returns:
        Same as `poly`, but with attributes cleaned up.

    Examples:
        >>> q0, _ = numpoly.align_polynomials(*numpoly.variable(2))
        >>> q0.indeterminants
        polynomial([q0, q1])
        >>> q0.exponents
        array([[0, 1],
               [1, 0]], dtype=uint32)
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
        exponents: numpy.typing.ArrayLike,
        coefficients: Sequence[numpy.typing.ArrayLike],
        names: Union[None, str, Tuple[str, ...], ndpoly] = None,
        retain_coefficients: Optional[bool] = None,
        retain_names: Optional[bool] = None,
) -> Tuple[numpy.ndarray, List[numpy.ndarray], Optional[Tuple[str, ...]]]:
    """
    Clean up polynomial attributes.

    Args:
        exponents:
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is
            the number of dimensions.
        coefficients:
            The polynomial coefficients. Must correspond to `exponents` by
            having the same length ``N``.
        names:
            The indeterminant names, either as string names or as
            simple polynomials. Must correspond to the exponents by having
            length ``D``.
        retain_coefficients:
            Do not remove redundant coefficients. If omitted use global
            defaults.
        retain_names:
            Do not remove redundant names. If omitted use global defaults.

    Returns:
        Same as inputs `exponents`, `coefficients` and `names`, but
        post-processed.

    """
    exponents = numpy.asarray(exponents)
    if exponents.ndim != 2:
        raise PolynomialConstructionError(
            f"expected exponents.ndim == 2; found {exponents.ndim}")

    coefficients_ = [numpy.asarray(coefficient)
                     for coefficient in coefficients]
    if coefficients_ and len(exponents) != len(coefficients_):
        raise PolynomialConstructionError(
            "expected len(exponents) == len(coefficients_); "
            f"found {len(exponents)} != {len(coefficients_)}")

    if retain_coefficients is None:
        retain_coefficients = numpoly.get_options()["retain_coefficients"]
    if not retain_coefficients and coefficients_:
        exponents, coefficients_ = remove_redundant_coefficients(
            exponents, coefficients_)

    if isinstance(names, numpoly.ndpoly):
        names = names.names
    if isinstance(names, str):
        if exponents.shape[1] > 1:
            names = tuple(f"{names}{idx}"
                          for idx in range(exponents.shape[1]))
        else:
            names = (names,)
    if names:
        if len(names) != exponents.shape[1]:
            raise PolynomialConstructionError(
                "Name length incompatible exponent length; "
                f"len({names}) != {exponents.shape[1]}")
        if sorted(set(names)) != sorted(names):
            raise PolynomialConstructionError(
                f"Duplicate indeterminant names: {names}")

    if retain_names is None:
        retain_names = numpoly.get_options()["retain_names"]
    if not retain_names:
        exponents, names = remove_redundant_names(exponents, names)

    exponents_, count = numpy.unique(exponents, return_counts=True, axis=0)
    if numpy.any(count > 1):
        raise PolynomialConstructionError(
            f"Duplicate exponent keys found: {exponents_[count > 1][0]}")

    return numpy.asarray(exponents), list(coefficients_), names


def remove_redundant_coefficients(
        exponents: numpy.typing.ArrayLike,
        coefficients: Sequence[numpy.typing.ArrayLike],
) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
    """
    Remove coefficients if they are redundant to the polynomial representation.

    Will always keep at least one coefficient.

    Args:
        exponents:
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is
            the number of dimensions.
        coefficients:
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
        >>> remove_redundant_coefficients([[0]], [[]])
        (array([[0]]), [array([], dtype=float64)])

    """
    exponents_ = numpy.asarray(exponents)
    coefficients_ = [
        numpy.asarray(coefficient) for coefficient in coefficients]
    assert len(exponents_) == len(coefficients_), (exponents, coefficients)

    elements = list(zip(*[
        (exponent, coefficient)
        for exponent, coefficient in zip(exponents_, coefficients_)
        if numpy.any(coefficient) or not numpy.any(exponent)
    ]))
    if not elements:
        exponents_ = numpy.zeros((1, exponents_.shape[-1]), dtype="uint32")
        coefficients_ = [numpy.zeros_like(coefficients_[0])]

    else:
        exponents_ = numpy.asarray(elements[0], dtype=int)
        coefficients_ = list(elements[1])
    return exponents_, coefficients_


def remove_redundant_names(
        exponents: numpy.typing.ArrayLike,
        names: Optional[Sequence[str]],
) -> Tuple[numpy.ndarray, Optional[Tuple[str, ...]]]:
    """
    Remove names if they are redundant to the polynomial representation.

    Will always keep at least one dimension.

    Args:
        exponents:
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is
            the number of dimensions.
        names:
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
        ('q0', 'q1')
        >>> exponents, names = remove_redundant_names(
        ...     [[0, 0], [0, 1]], ["q0", "q1"])
        >>> exponents
        array([[0],
               [1]])
        >>> names
        ('q1',)
        >>> exponents, names = remove_redundant_names(
        ...     [[0, 0]], ["q0", "q1"])
        >>> exponents
        array([[0]])
        >>> names
        ('q0',)

    """
    exponents_ = numpy.asarray(exponents)
    indices = numpy.any(exponents_ != 0, 0)
    if not numpy.any(indices):
        indices[0] = True

    exponents_ = exponents_[:, indices]
    if names is None:
        return exponents_, names
    names_ = tuple(numpy.array(names)[indices].tolist())
    return exponents_, names_
