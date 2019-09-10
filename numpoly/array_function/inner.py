import numpy
import numpoly

from .implements import implements


@implements(numpy.inner)
def inner(a, b):
    a, b = numpoly.align_polynomial_exponents(a, b)
    return numpoly.sum(numpoly.multiply(a, b), axis=-1)
