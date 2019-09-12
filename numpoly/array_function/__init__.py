"""
collection of numpy wrapper functions.

The numpy library comes with a large array of functions for manipulation of
numpy.ndarray objects. Many of these functions are supported in Numpoly as
well.

For numpy version >=1.17, the numpy library started to support dispatching
functionality to subclasses. This means that the functions in numpoly with the
same name as a numpy counterpart will work irrespectively if the function used
was from numpy or numpoly.

For example::

    >>> poly = numpoly.symbols("x")**numpy.arange(4)
    >>> print(poly)
    [1 x x**2 x**3]
    >>> print(numpoly.sum(poly, keepdims=True))
    [1+x+x**2+x**3]
    >>> print(numpy.sum(poly, keepdims=True)) # doctest: +SKIP
    [1+x+x**2+x**3]

Not everything is possible to support, and for the functions that are
supported, not all arguments are supportable.
"""
from .common import ARRAY_FUNCTIONS

from .absolute import absolute as abs, absolute
from .add import add
from .any import any
from .all import all
from .allclose import allclose
from .around import around as round, around
from .array_repr import array_repr
from .array_str import array_str
from .common_type import common_type
from .concatenate import concatenate
from .cumsum import cumsum
from .divide import divide
from .equal import equal
from .floor_divide import floor_divide
from .inner import inner
from .isclose import isclose
from .isfinite import isfinite
from .logical_and import logical_and
from .logical_or import logical_or
from .mean import mean
from .moveaxis import moveaxis
from .multiply import multiply
from .negative import negative
from .not_equal import not_equal
from .outer import outer
from .positive import positive
from .power import power
from .prod import prod
from .rint import rint
from .square import square
from .subtract import subtract
from .sum import sum
