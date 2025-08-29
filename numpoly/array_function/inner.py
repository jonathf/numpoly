"""Inner product of two arrays."""

from __future__ import annotations

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.inner)
def inner(a: PolyLike, b: PolyLike) -> ndpoly:
    """
    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    """
    a, b = numpoly.align_exponents(a, b)
    return numpoly.sum(numpoly.multiply(a, b), axis=-1)
