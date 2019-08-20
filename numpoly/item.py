"""
Polynomial get item::

    >>> from numpoly import polynomial
    >>> poly = polynomial({(0,): [[1., 2.], [3., 4.]], (1,): [[4., 5.], [6., 7.]]})
    >>> print(poly)
    [[1.0+4.0q0 2.0+5.0q0]
     [3.0+6.0q0 4.0+7.0q0]]
    >>> print(poly["1"])
    [[4. 5.]
     [6. 7.]]
    >>> print(poly[0])
    [1.0+4.0q0 2.0+5.0q0]
    >>> print(poly[:, 1])
    [2.0+5.0q0 4.0+7.0q0]
"""
import numpy

from . import construct


def getitem(poly, index):
    # for diagnostic purposes, allow to access coefficients directly through
    # string items
    if isinstance(index, str):
        out = super(poly.__class__, poly).__getitem__(index)
        return numpy.asarray(out)

    if isinstance(index, tuple):
        index = (slice(None),) + index
    else:
        index = (slice(None), index)
    out = construct.polynomial_from_attributes(
        poly.exponents, numpy.array(poly.coefficients)[index])
    return out


def setitem(poly, index, value):
    raise NotImplementedError("missing")
