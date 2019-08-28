"""Evaluate polynomial."""
from itertools import product
import numpy
import numpoly

from . import array_function


def call(poly, *args, **kwargs):
    """
    Evaluate polynomial by inserting new values in to the indeterminants.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to evaluate.
        args (int, float, numpy.ndarray, numpoly.ndpoly):
            Argument to evaluate indeterminants. Ordered positional by
            ``poly.indeterminants``.
        kwargs (int, float, numpy.ndarray, numpoly.ndpoly):
            Same as ``args``, but positioned by name.

    Returns:
        (numpoly.ndpoly):
            Evaluated polynomial.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([[x, x-1], [y, y+x]])
        >>> print(poly)
        [[x -1+x]
         [y y+x]]
        >>> print(poly(1, 0))
        [[1 0]
         [0 1]]
        >>> print(poly(1, y=[0, 1, 2]))
        [[[1 1 1]
          [0 0 0]]
        <BLANKLINE>
         [[0 1 2]
          [1 2 3]]]
        >>> print(poly(y=x-1, x=2*y))
        [[2*y -1+2*y]
         [-1+x -1+2*y+x]]
    """
    # Make sure kwargs contains all args and nothing but indeterminants:
    for arg, indeterminant in zip(args, poly._indeterminants):
        if indeterminant in kwargs:
            raise TypeError(
                "multiple values for argument '%s'" % indeterminant)
        kwargs[indeterminant] = arg
    extra_args = [key for key in kwargs if key not in poly._indeterminants]
    if extra_args:
        raise TypeError("unexpected keyword argument '%s'" % extra_args[0])

    if not kwargs:
        return poly.copy()

    # Saturate kwargs with values not given:
    for indeterminant in poly.indeterminants:
        name = indeterminant._indeterminants[0]
        if name not in kwargs:
            kwargs[name] = indeterminant

    # There can only be one shape:
    ones = numpy.ones((), dtype=int)
    for value in kwargs.values():
        ones = ones * numpy.ones(numpoly.polynomial(value).shape, dtype=int)

    # main loop:
    out = 0
    for exponent, coefficient in zip(poly.exponents, poly.coefficients):
        term = ones
        for power, name in zip(exponent, poly._indeterminants):
            term = term*kwargs[name]**power
        shape = coefficient.shape+ones.shape
        out = out+array_function.outer(coefficient, term).reshape(shape)

    return numpoly.polynomial(out)
