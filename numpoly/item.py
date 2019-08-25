"""
Polynomial get item::

    >>> x, y = numpoly.symbols("x y")
    >>> poly = numpoly.polynomial([[1-4*x, x**2], [y-3, x*y*y]])
    >>> print(poly)
    [[1-4*x x**2]
     [-3+y x*y**2]]
    >>> print(poly[0])
    [1-4*x x**2]
    >>> print(poly[:, 1])
    [x**2 x*y**2]

Developer access using string::

    >>> print([str(exponent) for exponent in poly._exponents])
    ['00', '01', '10', '12', '20']
    >>> print(poly["10"])
    [[-4  0]
     [ 0  0]]
"""
import numpy
import numpoly


def getitem(poly, index):
    # for diagnostic purposes, allow to access coefficients directly through
    # string items
    if isinstance(index, (str, unicode)):
        out = super(poly.__class__, poly).__getitem__(index)
        return numpy.asarray(out)

    if isinstance(index, tuple):
        index = (slice(None),) + index
    else:
        index = (slice(None), index)

    return numpoly.polynomial_from_attributes(
        exponents=poly.exponents,
        coefficients=numpy.array(poly.coefficients)[index],
        indeterminants=poly._indeterminants,
    )


def setitem(poly, index, value):
    raise NotImplementedError("missing")
