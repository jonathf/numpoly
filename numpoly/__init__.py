# pylint: disable=wildcard-import
"""Numpoly -- Multivariate polynomials as numpy elements."""
from .baseclass import ndpoly

from .align import (
    align_polynomials,
    align_exponents,
    align_indeterminants,
    align_shape,
    align_dtype,
)
from .construct import (
    polynomial,
    aspolynomial,
    clean_attributes,
)
from .sympy_ import to_sympy

from .array_function import *
from .poly_function import *
