"""Inner product of two arrays."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.inner)
def inner(a, b):
    """
    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    """
    a, b = numpoly.align_exponents(a, b)
    return numpoly.sum(numpoly.multiply(a, b), axis=-1)
