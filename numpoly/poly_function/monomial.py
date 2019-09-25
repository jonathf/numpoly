"""Monomial construction."""
from __future__ import division

import numpy
import numpoly


def monomial(indeterminants, start, stop=None, ordering="G", cross_truncation=1.):
    """
    Create an polynomial monomial expansion.

    Args:
        indeterminants (numpoly.ndpoly, str, Tuple[str, ...])
            The indeterminants used to create the monomials expansion.
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
            ``monomial(("x", "y"), start=0, stop=2, ordering=ordering)`` we
            get:

            ========  ==================
            ordering  output
            ========  ==================
            ""        [1 y y^2 x xy x^2]
            "G"       [1 y x y^2 xy x^2]
            "I"       [x^2 xy x y^2 y 1]
            "R"       [1 x x^2 y xy y^2]
            "GIR"     [y^2 xy x^2 y x 1]
            ========  ==================
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Returns:
        (numpoly.ndpoly):
            Monomial expansion.

    Examples:
        >>> print(numpoly.monomial("x", 4))
        [1 x x**2 x**3 x**4]
        >>> print(numpoly.monomial(("x", "y"), 4, 4, ordering="GR"))
        [x**4 x**3*y y**4 x**2*y**2 x*y**3]
        >>> print(numpoly.monomial(("x", "y"), [1, 1], [2, 2]))
        [x*y x*y**2 x**2*y x**2*y**2]

    """
    if stop is None:
        start, stop = numpy.array(0), start

    start = numpy.array(start, dtype=int)
    stop = numpy.array(stop, dtype=int)
    dimensions = max(start.size, stop.size, len(indeterminants))

    indices = bindex(
        start=numpy.min(start),
        stop=2*numpy.max(stop),
        dimensions=dimensions,
        ordering=ordering,
        cross_truncation=cross_truncation,
    )

    if start.size == 1:
        below = numpy.sum(indices, -1) >= start
    else:
        start = numpy.ones(dimensions, dtype=int)*start
        below = numpy.all(indices-start >= 0, -1)

    if stop.size == 1:
        above = numpy.sum(indices, -1) <= stop.item()
    else:
        stop = numpy.ones(dimensions, dtype=int)*stop
        above = numpy.all(stop-indices >= 0, -1)
    indices = indices[above*below]

    poly = numpoly.ndpoly(
        exponents=indices,
        shape=(len(indices),),
        indeterminants=indeterminants,
    )
    for coeff, key in zip(
            numpy.eye(len(indices), dtype=int), poly.keys):
        poly[key] = coeff
    return poly


def bindex(start, stop=None, dimensions=1, ordering="G", cross_truncation=1.):
    """
    Generate multi-indices for the monomial exponents.

    Args:
        start (Union[int, numpy.ndarray]):
            The lower order of the indices. If array of int, counts as lower
            bound for each axis.
        stop (Union[int, numpy.ndarray, None]):
            The maximum shape included. If omitted: stop <- start; start <- 0
            If int is provided, set as largest total order. If array of int,
            set as upper bound for each axis.
        dimensions (int):
            The number of dimensions in the expansion
        ordering (str):
            Criteria to sort the indices by.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion. Ignored if ``stop`` is a array.

    Returns:
        list:
            Order list of indices.

    Examples:
        >>> print(bindex(2, 3, 2).tolist())
        [[0, 2], [1, 1], [2, 0], [0, 3], [1, 2], [2, 1], [3, 0]]
        >>> print(bindex(2, [1, 3], 2, cross_truncation=0).tolist())
        [[0, 2], [1, 1], [0, 3], [1, 2], [1, 3]]
        >>> print(bindex([1, 2], [2, 3], 2, cross_truncation=0).tolist())
        [[1, 2], [1, 3], [2, 2], [2, 3]]
        >>> print(bindex(1, 3, 2, cross_truncation=0).tolist())  # doctest: +NORMALIZE_WHITESPACE
        [[0, 1], [1, 0], [0, 2], [1, 1], [2, 0], [0, 3], [1, 2], [2, 1],
            [3, 0], [1, 3], [2, 2], [3, 1], [2, 3], [3, 2], [3, 3]]
        >>> print(bindex(1, 3, 2, cross_truncation=1).tolist())  # doctest: +NORMALIZE_WHITESPACE
        [[0, 1], [1, 0], [0, 2], [1, 1], [2, 0],
            [0, 3], [1, 2], [2, 1], [3, 0]]
        >>> print(bindex(1, 3, 2, cross_truncation=1.5).tolist())
        [[0, 1], [1, 0], [0, 2], [1, 1], [2, 0], [0, 3], [3, 0]]
        >>> print(bindex(1, 3, 2, cross_truncation=2).tolist())
        [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0]]
        >>> print(bindex(0, 1, 3).tolist())
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
        >>> print(bindex(  # doctest: +NORMALIZE_WHITESPACE
        ...     [1, 1], 3, 2, cross_truncation=0).tolist())
        [[1, 1], [1, 2], [2, 1], [1, 3], [2, 2],
            [3, 1], [2, 3], [3, 2], [3, 3]]

    """
    if stop is None:
        start, stop = 0, start
    start = numpy.array(start, dtype=int).flatten()
    stop = numpy.array(stop, dtype=int).flatten()
    ordering = ordering.upper()
    start[start < 0] = 0

    indices = _bindex(start, stop, dimensions, cross_truncation)
    if "G" in ordering:
        indices = indices[numpy.argsort(numpy.sum(indices, -1))]

    if "I" in ordering:
        indices = indices[::-1]

    if "R" in ordering:
        indices = indices[:, ::-1]

    return indices


def _bindex(start, stop, dimensions=1, cross_truncation=1.):
    """Backend for the bindex function."""
    # At the beginning the current list of indices just ranges over the
    # last dimension.
    bound = stop.max()+1
    range_ = numpy.arange(bound, dtype=int)
    indices = range_[:, numpy.newaxis]

    for _ in range(dimensions-1):

        # Repeats the current set of indices.
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = numpy.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices)//bound)[:, numpy.newaxis]

        # Puts them two together.
        indices = numpy.column_stack((front, indices))

        # Truncate at each iteration to ensure memory usage is low enough
        if stop.size == 1 and cross_truncation > 0:
            lhs = numpy.sum(indices**(1/cross_truncation), -1)
            rhs = numpy.max(stop, -1)**(1/cross_truncation)
            indices = indices[lhs <= rhs]
        else:
            indices = indices[numpy.all(indices <= stop, -1)]

    if start.size == 1:
        indices = indices[numpy.sum(indices, -1) >= start.item()]
    else:
        indices = indices[numpy.all(indices >= start, -1)]
    return indices.reshape(-1, dimensions)