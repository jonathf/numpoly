"""Align polynomials."""
from __future__ import annotations
from typing import Tuple

import numpy
import numpoly
from .baseclass import ndpoly, PolyLike


def align_polynomials(*polys: PolyLike) -> Tuple[ndpoly, ...]:
    """
    Align polynomial such that dimensionality, shape, etc. are compatible.

    Alignment includes shape (for broadcasting), indeterminants, exponents and
    dtype.

    Args:
        polys:
            Polynomial to make adjustment to.

    Returns:
        Same as ``polys``, but internal adjustments made to make them
        compatible for further operations.

    Examples:
        >>> q0 = numpoly.variable()
        >>> q0q1 = numpoly.variable(2)
        >>> q0
        polynomial(q0)
        >>> q0.coefficients
        [1]
        >>> q0.indeterminants
        polynomial([q0])
        >>> q0, _ = numpoly.align_polynomials(q0, q0q1)
        >>> q0
        polynomial([q0, q0])
        >>> q0.coefficients
        [array([0, 0]), array([1, 1])]
        >>> q0.indeterminants
        polynomial([q0, q1])

    """
    # polys = numpoly.broadcast_arrays(*polys)
    polys = align_shape(*polys)
    # polys = align_indeterminants(*polys)
    polys = align_exponents(*polys)
    return polys


def align_shape(*polys: PolyLike) -> Tuple[ndpoly, ...]:
    """
    Align polynomial by shape.

    Args:
        polys:
            Polynomial to make adjustment to.

    Returns:
        Same as ``polys``, but internal adjustments made to make them
        compatible for further operations.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly1 = 4*q0
        >>> poly2 = numpoly.polynomial([[2*q0+1, 3*q0-q1]])
        >>> poly1.shape
        ()
        >>> poly2.shape
        (1, 2)
        >>> poly1, poly2 = numpoly.align_shape(poly1, poly2)
        >>> poly1
        polynomial([[4*q0, 4*q0]])
        >>> poly2
        polynomial([[2*q0+1, -q1+3*q0]])
        >>> poly1.shape
        (1, 2)
        >>> poly2.shape
        (1, 2)

    """
    # return tuple(numpoly.broadcast_arrays(*polys))
    polys_ = [numpoly.aspolynomial(poly) for poly in polys]
    common = numpy.ones(
        numpy.broadcast_shapes(*[poly.shape for poly in polys_]), dtype=int)

    for idx, poly in enumerate(polys_):
        if poly.shape != common.shape:
            polys_[idx] = poly.from_attributes(
                exponents=poly.exponents,
                coefficients=tuple(coeff*common for coeff in poly.coefficients),
                names=poly.indeterminants,
            )
    return tuple(polys_)


def align_indeterminants(*polys: PolyLike) -> Tuple[ndpoly, ...]:
    """
    Align polynomial by indeterminants.

    Args:
        polys:
            Polynomial to make adjustment to.

    Returns:
        Same as ``polys``, but internal adjustments made to make them
        compatible for further operations.

    Examples:
        >>> q0 = numpoly.variable()
        >>> poly1 = numpoly.polynomial(2*q0+1)
        >>> q0, q1 = numpoly.variable(2)
        >>> poly2 = numpoly.polynomial(3*q0-q1)
        >>> poly1.indeterminants
        polynomial([q0])
        >>> poly2.indeterminants
        polynomial([q0, q1])
        >>> poly1, poly2 = numpoly.align_indeterminants(poly1, poly2)
        >>> poly1
        polynomial(2*q0+1)
        >>> poly2
        polynomial(-q1+3*q0)
        >>> poly1.indeterminants
        polynomial([q0, q1])
        >>> poly2.indeterminants
        polynomial([q0, q1])

    """
    polys_ = [numpoly.aspolynomial(poly) for poly in polys]
    common_names = tuple(sorted({
        str(name) for poly in polys_ for name in poly.names}))
    if not common_names:
        return tuple(polys_)

    for idx, poly in enumerate(polys_):
        if poly.names == common_names:
            continue
        indices = numpy.array([
            common_names.index(name)
            for name in poly.names
            if name in common_names
        ])
        exponents = numpy.zeros(
            (len(poly.keys), len(common_names)), dtype=int)
        if indices.size:
            exponents[:, indices] = poly.exponents
        polys_[idx] = numpoly.ndpoly.from_attributes(
            exponents=exponents,
            coefficients=poly.coefficients,
            names=common_names,
            retain_coefficients=True,
            retain_names=True,
        )
    return tuple(polys_)


def align_exponents(*polys: PolyLike) -> Tuple[ndpoly, ...]:
    """
    Align polynomials such that the exponents are the same.

    Aligning exponents assumes that the indeterminants is also aligned.

    Args:
        polys:
            Polynomial to make adjustment to.

    Returns:
        Same as ``polys``, but internal adjustments made to make them
        compatible for further operations.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly1 = q0*q1
        >>> poly2 = numpoly.polynomial([q0**5, q1**3-1])
        >>> poly1.exponents
        array([[1, 1]], dtype=uint32)
        >>> poly2.exponents
        array([[0, 0],
               [0, 3],
               [5, 0]], dtype=uint32)
        >>> poly1, poly2 = numpoly.align_exponents(poly1, poly2)
        >>> poly1
        polynomial(q0*q1)
        >>> poly2
        polynomial([q0**5, q1**3-1])
        >>> poly1.exponents
        array([[0, 0],
               [0, 3],
               [1, 1],
               [5, 0]], dtype=uint32)
        >>> poly2.exponents
        array([[0, 0],
               [0, 3],
               [1, 1],
               [5, 0]], dtype=uint32)

    """
    polys_ = [numpoly.aspolynomial(poly) for poly in polys]
    if not all(polys_[0].names == poly.names for poly in polys_):
        polys_ = list(align_indeterminants(*polys_))

    global_exponents = numpy.vstack([poly.exponents for poly in polys_])
    global_exponents = numpy.unique(global_exponents, axis=0).tolist()

    for idx, poly in enumerate(polys_):
        if numpy.all(poly.exponents == global_exponents):
            continue
        lookup = {
            tuple(exponent): coefficient
            for exponent, coefficient in zip(
                poly.exponents, poly.coefficients)
        }

        zeros = numpy.zeros(poly.shape, dtype=poly.dtype)
        coefficients = [lookup.get(tuple(exponent), zeros)
                        for exponent in global_exponents]
        polys_[idx] = poly.from_attributes(
            exponents=global_exponents,
            coefficients=coefficients,
            names=poly.names,
            retain_coefficients=True,
            retain_names=True,
        )
    return tuple(polys_)
