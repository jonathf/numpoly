"""Testing functions used for numpoly only functionality."""
from pytest import raises
import numpy
import numpoly
import sympy

X, Y = numpoly.symbols("X Y")


def test_numpoly_call():
    poly = X+Y
    with raises(TypeError):
        poly(1, X=2)
    with raises(TypeError):
        poly(1, 2, Y=3)
    with raises(TypeError):
        poly(not_an_arg=45)


def test_numpoly_ndpoly():
    poly = numpoly.ndpoly(exponents=[(1,)], shape=(), indeterminants="X")
    poly["1"] = 1
    assert poly == X
    poly = numpoly.ndpoly(exponents=[(1,)], shape=(), indeterminants=X)
    poly["1"] = 1
    assert poly == X
    poly = numpoly.ndpoly(
        exponents=[(1, 0), (0, 1)], shape=(), indeterminants=("X" ,"Y"))
    poly["10"] = 2
    poly["01"] = 3
    assert poly == 2*X+3*Y


def test_numpoly_polynomial():
    assert numpoly.polynomial() == 0
    assert numpoly.polynomial({(0,): 4}) == 4
    assert numpoly.polynomial(X) == X
    assert numpoly.polynomial(numpy.array((3,), dtype=[("0", int)])) == 3
    assert numpoly.polynomial(5.5) == 5.5
    assert numpoly.polynomial(sympy.symbols("X")) == X
    assert numpy.all(numpoly.polynomial([1, 2, 3]) == [1, 2, 3])
