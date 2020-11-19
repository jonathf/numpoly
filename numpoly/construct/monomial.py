"""Monomial construction."""
from __future__ import division

import numpy
import numpoly


def monomial(start, stop=None, dimensions=None, cross_truncation=1.,
             graded=False, reverse=False, allocation=None):
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
        dimensions (None, numpoly.ndpoly, int, str, Tuple[str, ...])
            The indeterminants names used to create the monomials expansion.
            If int provided, set the number of names to be used.
        cross_truncation (float):
            The hyperbolic cross truncation scheme to reduce the number of
            terms in expansion. In other words, it sets the :math:`q` in the
            :math:`L_q`-norm.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**3*q1*q2``,
            ``q0*q1**2*q2`` and ``q0*q1*q2**2``, which all have exponent
            sum of 5.
        reverse (bool):
            Reverse lexicographical sorting meaning that ``q0*q1**3`` is
            considered bigger than ``q0**3*q1``, instead of the opposite.
        allocation (Optional[int]):
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Returns:
        (numpoly.ndpoly):
            Monomial expansion.

    Examples:
        >>> numpoly.monomial(4)
        polynomial([1, q0, q0**2, q0**3])
        >>> numpoly.monomial(start=4, stop=5, dimensions=2,
        ...                  graded=True, reverse=True)
        polynomial([q1**4, q0*q1**3, q0**2*q1**2, q0**3*q1, q0**4])
        >>> numpoly.monomial(2, [3, 4], graded=True)
        polynomial([q0**2, q0*q1, q1**2, q1**3])
        >>> numpoly.monomial(
        ...     start=0,
        ...     stop=4,
        ...     dimensions=("q2", "q4"),
        ...     cross_truncation=0.5,
        ...     graded=True,
        ...     reverse=True,
        ... )
        polynomial([1, q4, q2, q4**2, q2**2, q4**3, q2**3])

    """
    if stop is None:
        start, stop = numpy.array(0), start

    start = numpy.array(start, dtype=int)
    stop = numpy.array(stop, dtype=int)
    if isinstance(dimensions, str):
        names, dimensions = (dimensions,), 1
    elif isinstance(dimensions, int):
        dimensions = max(start.size, stop.size, dimensions)
        names = numpoly.variable(dimensions).names
    elif dimensions is None:
        dimensions = max(start.size, stop.size)
        names = numpoly.variable(dimensions).names
    else:
        names, dimensions = dimensions, len(dimensions)

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
        allocation=allocation,
    )
    for coeff, key in zip(
            numpy.eye(len(indices), dtype=int), poly.keys):
        numpy.ndarray.__setitem__(poly, key, coeff)
    return poly
