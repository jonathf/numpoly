"""Test for `numpoly.construct`."""
from pytest import raises
import numpy

import sympy
import numpoly
from numpoly.construct.clean import (
    postprocess_attributes, PolynomialConstructionError)

XY = numpoly.variable(2)
X, Y = numpoly.symbols("q0"), numpoly.symbols("q1")


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
    with raises(PolynomialConstructionError):  # exponents len incompatible with name length
        postprocess_attributes([[1], [2]], [1, 2], ["x", "y", "z"])
    with raises(PolynomialConstructionError):  # duplicate names
        postprocess_attributes([[1, 1]], [1], ["x", "x"])


def test_aspolynomial():
    poly = 2*X-Y+1
    assert poly == numpoly.aspolynomial(poly)
    assert poly == numpoly.aspolynomial(poly, names=XY)
    assert poly == numpoly.aspolynomial(poly.todict(), names=XY)
    assert poly == numpoly.aspolynomial(poly, names=("q0", "q1"))
    with numpoly.global_options(
            varname_filter=r"\w+", force_number_suffix=False):
        assert numpy.all(
            numpoly.symbols("Z:2") == numpoly.aspolynomial(XY, names="Z"))
    assert poly == numpoly.aspolynomial(poly.todict(), names=("q0", "q1"))
    assert poly != numpoly.aspolynomial(poly.todict(), names=("q1", "q0"))
    assert X == numpoly.aspolynomial(Y, names="q0")
    assert poly != numpoly.aspolynomial(poly.todict(), names="q0")
    assert isinstance(numpoly.aspolynomial([1, 2, 3]), numpoly.ndpoly)
    assert numpy.all(numpoly.aspolynomial([1, 2, 3]) == [1, 2, 3])


def test_monomial():
    assert not numpoly.monomial(0).size
    assert numpoly.monomial(1) == 1
    assert numpy.all(numpoly.monomial(2, dimensions="q0") == [1, X])
    assert numpoly.monomial(1, 2, dimensions="q0") == X
    assert numpoly.monomial(1, 2, dimensions=None) == X


def test_polynomial():
    assert numpoly.polynomial() == 0
    assert numpoly.polynomial({(0,): 4}) == 4
    assert numpoly.polynomial({(1,): 5}, names="q0") == 5*X
    assert numpoly.polynomial(
        {(0, 1): 2, (1, 0): 3}, names=("q0", "q1")) == 3*X+2*Y
    with numpoly.global_options(varname_filter=r"\w+"):
        assert numpy.all(numpoly.polynomial(
            {(0, 1): [0, 1], (1, 0): [1, 0]}, names="Q"
        ) == numpoly.symbols("Q0 Q1"))
    assert numpoly.polynomial(X) == X
    assert numpoly.polynomial(numpy.array((3,), dtype=[(";", int)])) == 3
    assert numpoly.polynomial(5.5) == 5.5
    assert numpoly.polynomial(sympy.symbols("q0")) == X
    assert numpy.all(numpoly.polynomial([1, 2, 3]) == [1, 2, 3])
    assert numpy.all(numpoly.polynomial([[1, 2], [3, 4]]) == [[1, 2], [3, 4]])
    assert numpy.all(numpoly.polynomial(
        numpy.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]])


def test_symbols():
    assert numpoly.symbols() == X
    assert numpoly.symbols().shape == ()
    assert numpoly.symbols("q0") == X
    assert numpoly.symbols("q0").shape == ()
    assert numpoly.symbols("q0,") == X
    assert numpoly.symbols("q0,").shape == (1,)
    assert numpoly.symbols("q0", asarray=True) == X
    assert numpoly.symbols("q0", asarray=True).shape == (1,)

    assert numpoly.symbols("q:1").names == ("q0",)
    assert numpoly.symbols("q1").names == ("q1",)
