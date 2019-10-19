"""Test for `numpoly.construct`."""
from pytest import raises
import numpy

from numpoly.construct.clean import (
    postprocess_attributes, PolynomialConstructionError)


def test_postprocess_attributes():
    """Test related to polynomial construction from attributes."""
    with raises(PolynomialConstructionError):  # exponent.ndim too small
        postprocess_attributes(numpy.arange(2), [1, 2])
    with raises(PolynomialConstructionError):  # exponent.ndim too large
        postprocess_attributes(numpy.arange(24).reshape(2, 3, 4), [1, 2])
    with raises(PolynomialConstructionError):  # exponents len incompatible with coefficients
        postprocess_attributes(numpy.arange(6).reshape(2, 3), [1, 2, 3])
    with raises(PolynomialConstructionError):  # duplicate exponents
        postprocess_attributes([[1], [1]], [1, 2])
    with raises(PolynomialConstructionError):  # exponents len incompatible with indeterminants
        postprocess_attributes([[1], [2]], [1, 2], ["x", "y", "z"])
    with raises(PolynomialConstructionError):  # duplicate indeterminant names
        postprocess_attributes([[1], [2]], [1, 2], ["x", "x"])
