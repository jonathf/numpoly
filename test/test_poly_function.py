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
    poly = numpoly.ndpoly(
        exponents=[(1, 0), (0, 1)], shape=(2,), indeterminants="Q")
    poly["10"] = [1, 0]
    poly["01"] = [0, 1]
    assert numpy.all(poly == numpoly.symbols("Q0 Q1"))


def test_numpoly_polynomial():
    assert numpoly.polynomial() == 0
    assert numpoly.polynomial({(0,): 4}) == 4
    assert numpoly.polynomial({(1,): 5}, indeterminants="X") == 5*X
    assert numpoly.polynomial(
        {(0, 1): 2, (1, 0): 3}, indeterminants=("X", "Y")) == 3*X+2*Y
    assert numpy.all(numpoly.polynomial(
        {(0, 1): [0, 1], (1, 0): [1, 0]}, indeterminants="Q"
    ) == numpoly.symbols("Q0 Q1"))
    assert numpoly.polynomial(X) == X
    assert numpoly.polynomial(numpy.array((3,), dtype=[("0", int)])) == 3
    assert numpoly.polynomial(5.5) == 5.5
    assert numpoly.polynomial(sympy.symbols("X")) == X
    assert numpy.all(numpoly.polynomial([1, 2, 3]) == [1, 2, 3])
    assert numpy.all(numpoly.polynomial([[1, 2], [3, 4]]) == [[1, 2], [3, 4]])
    assert numpy.all(numpoly.polynomial(
        numpy.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]])


def test_numpoly_isconstant():
    assert not numpoly.polynomial(X).isconstant()
    assert numpoly.polynomial(1).isconstant()
    assert not numpoly.polynomial([1, X]).isconstant()
    assert numpoly.polynomial([1, 2]).isconstant()


def test_numpoly_tonumpy():
    assert isinstance(numpoly.tonumpy(numpoly.polynomial([1, 2, 3])), numpy.ndarray)
    with raises(ValueError):
        numpoly.tonumpy(X)
