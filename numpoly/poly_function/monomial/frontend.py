"""Monomial construction."""
from __future__ import division

import numpy
import numpoly

from .bindex import bindex


def monomial(start, stop=None, ordering="G", cross_truncation=1., names=None):
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
        ordering (str):
            The monomial ordering where the letters ``G``, ``I`` and ``R`` can
            be used to set grade, inverse and reverse to the ordering. For
            ``names=("x", "y"))`` we get for various values for
            ``ordering``:

            ========  =====================
            ordering  output
            ========  =====================
            ""        [1 y y**2 x x*y x**2]
            "G"       [1 y x y**2 x*y x**2]
            "I"       [x**2 x*y x y**2 y 1]
            "R"       [1 x x**2 y x*y y**2]
            "GIR"     [y**2 x*y x**2 y x 1]
            ========  =====================
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.
        names (None, numpoly.ndpoly, str, Tuple[str, ...])
            The indeterminants names used to create the monomials expansion.

    Returns:
        (numpoly.ndpoly):
            Monomial expansion.

    Examples:
        >>> numpoly.monomial(4)
        polynomial([1, q, q**2, q**3])
        >>> numpoly.monomial(4, 5, ordering="GR", names=("x", "y"))
        polynomial([x**4, x**3*y, x**2*y**2, x*y**3, y**4])
        >>> numpoly.monomial(2, [3, 4])
        polynomial([q1**2, q0*q1, q0**2, q1**3])
        >>> numpoly.monomial(0, 5, names=("x", "y"), cross_truncation=0.5)
        polynomial([1, y, x, y**2, x*y, x**2, y**3, x**3, y**4, x**4])

    """
    if stop is None:
        start, stop = numpy.array(0), start

    start = numpy.array(start, dtype=int)
    stop = numpy.array(stop, dtype=int)
    dimensions = 1 if names is None else len(names)
    dimensions = max(start.size, stop.size, dimensions)

    indices = bindex(
        start=start,
        stop=stop,
        dimensions=dimensions,
        ordering=ordering,
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
