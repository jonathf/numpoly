"""Monomial construction."""
from __future__ import division

import numpy
import numpoly


def monomial(start, stop=None, cross_truncation=1.,
             names=None, graded=False, reverse=False):
    """
    Create an polynomial monomial expansion.

    Args:
        start (int, numpy.ndarray):
            The minimum polynomial to include. If int is provided, set as
            lowest total order. If array of int, set as lower order along each
            indeterminant.
        stop (int, numpy.ndarray):
            The maximum shape included. If omitted:
            ``stop <- start; start <- 0`` If int is provided, set as largest
            total order. If array of int, set as largest order along each
            indeterminant.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.
        names (None, numpoly.ndpoly, str, Tuple[str, ...])
            The indeterminants names used to create the monomials expansion.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``x**2*y**2*z**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``x**3*y*z``, ``x*y**2*z``
            and ``x*y*z**2``, which all have exponent sum of 5.
        reverse (bool):
            Reverse lexicographical sorting meaning that ``x*y**3`` is
            considered bigger than ``x**3*y``, instead of the opposite.

    Returns:
        (numpoly.ndpoly):
            Monomial expansion.

    Examples:
        >>> numpoly.monomial(4)
        polynomial([1, q, q**2, q**3])
        >>> numpoly.monomial(4, 5, names=("x", "y"),
        ...                  graded=True, reverse=True)
        polynomial([y**4, x*y**3, x**2*y**2, x**3*y, x**4])
        >>> numpoly.monomial(2, [3, 4], graded=True)
        polynomial([q0**2, q0*q1, q1**2, q1**3])
        >>> numpoly.monomial(0, 5, names=("x", "y"),
        ...                  cross_truncation=0.5, graded=True, reverse=True)
        polynomial([1, y, x, y**2, x*y, x**2, y**3, x**3, y**4, x**4])

    """
    if stop is None:
        start, stop = numpy.array(0), start

    start = numpy.array(start, dtype=int)
    stop = numpy.array(stop, dtype=int)
    dimensions = 1 if names is None else len(names)
    dimensions = max(start.size, stop.size, dimensions)

    indices = numpoly.glexindex(
        start=start,
        stop=stop,
        dimensions=dimensions,
        graded=graded,
        reverse=reverse,
        cross_truncation=cross_truncation,
    )

    poly = numpoly.ndpoly(
        exponents=indices,
        shape=(len(indices),),
        names=names,
    )
    for coeff, key in zip(
            numpy.eye(len(indices), dtype=int), poly.keys):
        poly[key] = coeff
    return poly
