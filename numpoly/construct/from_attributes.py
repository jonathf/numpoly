"""Construct polynomial from polynomial attributes."""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

import numpy.typing

import numpoly

from . import clean
from ..baseclass import ndpoly


def polynomial_from_attributes(
        exponents: numpy.typing.ArrayLike,
        coefficients: Sequence[numpy.typing.ArrayLike],
        names: Union[None, str, Tuple[str, ...], ndpoly] = None,
        dtype: Optional[numpy.typing.DTypeLike] = None,
        allocation: Optional[int] = None,
        retain_coefficients: Optional[bool] = None,
        retain_names: Optional[bool] = None,
) -> ndpoly:
    """
    Construct polynomial from polynomial attributes.

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
        dtype:
            The data type of the polynomial. If omitted, extract from
            `coefficients`.
        allocation:
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.
        retain_coefficients:
            Do not remove redundant coefficients. If omitted use global
            defaults.
        retain_names:
            Do not remove redundant names. If omitted use global defaults.

    Returns:
        Polynomial array with attributes determined by the input.

    Examples:
        >>> numpoly.ndpoly.from_attributes(
        ...     exponents=[(0,), (1,)],
        ...     coefficients=[[1, 0], [0, 1]],
        ...     names="q4",
        ... )
        polynomial([1, q4])
        >>> numpoly.ndpoly.from_attributes(
        ...     exponents=[(0, 0, 0), (1, 1, 2)],
        ...     coefficients=[4, -1],
        ...     names=("q2", "q4", "q10"),
        ... )
        polynomial(-q2*q4*q10**2+4)
        >>> numpoly.ndpoly.from_attributes(
        ...     exponents=[(0,)],
        ...     coefficients=[0],
        ... )
        polynomial(0)

    """
    exponents, coefficients, names = clean.postprocess_attributes(
        exponents=exponents,
        coefficients=coefficients,
        names=names,
        retain_coefficients=retain_coefficients,
        retain_names=retain_names,
    )
    if coefficients:
        dtype = coefficients[0].dtype if dtype is None else dtype
        shape = coefficients[0].shape
    else:
        dtype = dtype if dtype else int
        shape = ()
    poly = numpoly.ndpoly(
        exponents=exponents,
        shape=shape,
        names=names,
        dtype=dtype,
        allocation=allocation,
    )
    for key, values in zip(poly.keys, coefficients):
        poly.values[key] = values
    return poly
