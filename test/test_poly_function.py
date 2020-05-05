"""Testing functions used for numpoly only functionality."""
from pytest import raises

import sympy
import numpy
import numpoly
from numpoly.poly_function.monomial.cross_truncation import cross_truncate

X, Y = numpoly.symbols("X Y")


def test_call():
    poly = X+Y
    with raises(TypeError):
        poly(1, X=2)
    with raises(TypeError):
        poly(1, 2, Y=3)
    with raises(TypeError):
        poly(not_an_arg=45)
    poly = numpoly.polynomial([2, X-Y+1])
    assert numpy.all(poly(X=Y) == [2, 1])


def test_ndpoly():
    poly = numpoly.ndpoly(exponents=[(1,)], shape=(), names="X")
    poly["<"] = 1
    assert poly == X
    poly = numpoly.ndpoly(exponents=[(1,)], shape=(), names=X)
    poly["<"] = 1
    assert poly == X
    poly = numpoly.ndpoly(
        exponents=[(1, 0), (0, 1)], shape=(), names=("X" ,"Y"))
    poly["<;"] = 2
    poly[";<"] = 3
    assert poly == 2*X+3*Y
    poly = numpoly.ndpoly(
        exponents=[(1, 0), (0, 1)], shape=(2,), names="Q")
    poly["<;"] = [1, 0]
    poly[";<"] = [0, 1]
    assert numpy.all(poly == numpoly.symbols("Q0 Q1"))


def test_polynomial():
    assert numpoly.polynomial() == 0
    assert numpoly.polynomial({(0,): 4}) == 4
    assert numpoly.polynomial({(1,): 5}, names="X") == 5*X
    assert numpoly.polynomial(
        {(0, 1): 2, (1, 0): 3}, names=("X", "Y")) == 3*X+2*Y
    assert numpy.all(numpoly.polynomial(
        {(0, 1): [0, 1], (1, 0): [1, 0]}, names="Q"
    ) == numpoly.symbols("Q0 Q1"))
    assert numpoly.polynomial(X) == X
    assert numpoly.polynomial(numpy.array((3,), dtype=[(";", int)])) == 3
    assert numpoly.polynomial(5.5) == 5.5
    assert numpoly.polynomial(sympy.symbols("X")) == X
    assert numpy.all(numpoly.polynomial([1, 2, 3]) == [1, 2, 3])
    assert numpy.all(numpoly.polynomial([[1, 2], [3, 4]]) == [[1, 2], [3, 4]])
    assert numpy.all(numpoly.polynomial(
        numpy.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]])


def test_poly_divide():
    const = numpoly.polynomial([1, 2])
    poly1 = numpoly.polynomial(X)
    poly2 = numpoly.polynomial([1, X])
    poly3 = numpoly.polynomial([[0, Y], [X, 1]])
    assert numpy.all(numpoly.poly_divide(poly1, const) == [X, 0.5*X])
    assert numpy.all(numpoly.poly_divide(const, poly1) == [0, 0])
    assert numpy.all(poly1.__truediv__(poly2) == [X, 1])
    assert numpy.all(poly2.__rtruediv__(poly3) == [[0, 0], [X, 0]])
    assert numpy.all(poly2.__div__(poly1) == [0, 1])
    assert numpy.all(poly3.__rdiv__(poly2) == [[0, 0], [0, X]])


def test_poly_divmod():
    assert numpoly.poly_divmod(X+3, X+2) == (1, 1)
    assert numpoly.poly_divmod(Y+3, X+2) == (0, Y+3)
    assert numpoly.poly_divmod(3, X+2) == (0, 3)
    assert numpoly.poly_divmod(X, X+2) == (1, -2)
    assert numpoly.poly_divmod(Y, X+2) == (0, Y)
    assert numpoly.poly_divmod(X*Y, X+2) == (Y, -2*Y)

    assert divmod(X+3, 2) == (0.5*X+1.5, 0)
    assert divmod(Y+3, 2) == (0.5*Y+1.5, 0)
    assert divmod(numpoly.polynomial(3), 2) == (1.5, 0)
    assert divmod(X, 2) == (0.5*X, 0)
    assert divmod(Y, 2) == (0.5*Y, 0)
    assert divmod(X*Y, 2) == (0.5*X*Y, 0)

    assert (X+3).__divmod__(X) == (1, 3)
    assert (Y+3).__divmod__(X) == (0, Y+3)
    with raises(numpoly.FeatureNotSupported):
        assert numpy.array(3).__divmod__(X) == (0, 3)
    assert X.__divmod__(X) == (1, 0)
    assert Y.__divmod__(X) == (0, Y)
    assert (X*Y).__divmod__(X) == (Y, 0)


def test_isconstant():
    assert not numpoly.polynomial(X).isconstant()
    assert numpoly.polynomial(1).isconstant()
    assert not numpoly.polynomial([1, X]).isconstant()
    assert numpoly.polynomial([1, 2]).isconstant()


def test_tonumpy():
    assert isinstance(numpoly.tonumpy(numpoly.polynomial([1, 2, 3])), numpy.ndarray)
    with raises(numpoly.FeatureNotSupported):
        numpoly.tonumpy(X)


def test_cross_truncate():
    indices = numpy.array(numpy.mgrid[:10, :10]).reshape(2, -1).T

    assert not numpy.any(cross_truncate(indices, -1, norm=0))
    assert numpy.all(indices[cross_truncate(indices, 0, norm=0)].T ==
                     [[0], [0]])
    assert numpy.all(indices[cross_truncate(indices, 1, norm=0)].T ==
                     [[0, 0, 1], [0, 1, 0]])
    assert numpy.all(indices[cross_truncate(indices, 2, norm=0)].T ==
                     [[0, 0, 0, 1, 2], [0, 1, 2, 0, 0]])

    assert not numpy.any(cross_truncate(indices, -1, norm=1))
    assert numpy.all(indices[cross_truncate(indices, 0, norm=1)].T ==
                     [[0], [0]])
    assert numpy.all(indices[cross_truncate(indices, 1, norm=1)].T ==
                     [[0, 0, 1], [0, 1, 0]])
    assert numpy.all(indices[cross_truncate(indices, 2, norm=1)].T ==
                     [[0, 0, 0, 1, 1, 2], [0, 1, 2, 0, 1, 0]])

    assert not numpy.any(cross_truncate(indices, -1, norm=100))
    assert numpy.all(indices[cross_truncate(indices, 0, norm=100)].T ==
                     [[0], [0]])
    assert numpy.all(indices[cross_truncate(indices, 1, norm=100)].T ==
                     [[0, 0, 1], [0, 1, 0]])
    assert numpy.all(indices[cross_truncate(indices, 2, norm=100)].T ==
                     [[0, 0, 0, 1, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1]])

    assert not numpy.any(cross_truncate(indices, -1, norm=numpy.inf))
    assert numpy.all(indices[cross_truncate(indices, 0, norm=numpy.inf)].T ==
                     [[0], [0]])
    assert numpy.all(indices[cross_truncate(indices, 1, norm=numpy.inf)].T ==
                     [[0, 0, 1, 1], [0, 1, 0, 1]])
    assert numpy.all(indices[cross_truncate(indices, 2, norm=numpy.inf)].T ==
                     [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])


def test_bindex():
    assert not numpoly.bindex(0).size
    assert numpy.all(numpoly.bindex(1) == [[0]])
    assert numpy.all(numpoly.bindex(5) ==
                     [[0], [1], [2], [3], [4]])
    assert numpy.all(numpoly.bindex(2, dimensions=2) ==
                     [[0, 0], [0, 1], [1, 0]])
    assert numpy.all(numpoly.bindex(start=2, stop=3, dimensions=2) ==
                     [[0, 2], [1, 1], [2, 0]])
    assert numpy.all(numpoly.bindex(start=2, stop=[3, 4], dimensions=2) ==
                     [[0, 2], [1, 1], [2, 0], [0, 3]])
    assert numpy.all(numpoly.bindex(start=[2, 5], stop=[3, 6], dimensions=2) ==
                     [[1, 1], [2, 0], [1, 2], [0, 5]])
    assert numpy.all(numpoly.bindex(start=2, stop=3, dimensions=2, ordering="I") ==
                     [[2, 0], [1, 1], [0, 2]])
    assert numpy.all(numpoly.bindex(start=2, stop=4, dimensions=2, cross_truncation=0) ==
                     [[0, 2], [2, 0], [0, 3], [3, 0]])
    assert numpy.all(numpoly.bindex(start=2, stop=4, dimensions=2, cross_truncation=1) ==
                     [[0, 2], [1, 1], [2, 0], [0, 3], [1, 2], [2, 1], [3, 0]])
    assert numpy.all(numpoly.bindex(start=2, stop=4, dimensions=2, cross_truncation=2) ==
                     [[0, 2], [1, 1], [2, 0], [0, 3], [1, 2], [2, 1], [3, 0], [2, 2]])
    assert numpy.all(numpoly.bindex(start=0, stop=2, dimensions=3) ==
                     [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])


def test_monomial():
    assert not numpoly.monomial(0).size
    assert numpoly.monomial(1) == 1
    assert numpy.all(numpoly.monomial(2, names="X") == [1, X])
    assert numpoly.monomial(1, 2, names="X") == X


def test_poly_remainder():
    poly = numpoly.polynomial([[1, 2*X], [3*Y+X, 4]])
    assert numpy.all(numpoly.poly_remainder(poly, 2) == [[0, 0], [0, 0]])
    assert numpy.all(numpy.array([1, 2]) % poly == [[0, 2], [1, 0]])
    assert numpy.all(poly % numpy.array([1, 2]) == 0)
    assert numpy.all(poly.__mod__(numpy.array([1, 2])) == 0)
    assert numpy.all(poly.__rmod__(numpy.array([1, 2])) == [[0, 2], [1, 0]])


def test_symbols():
    assert numpoly.symbols("q").names == ("q",)
    assert numpoly.symbols("q:1").names == ("q0",)
    assert numpoly.symbols("q1").names == ("q1",)
    numpoly.baseclass.INDETERMINANT_DEFAULTS["force_suffix"] = True
    assert numpoly.symbols("q").names == ("q0",)
    assert numpoly.symbols("q:1").names == ("q0",)
    assert numpoly.symbols("q1").names == ("q1",)
    numpoly.baseclass.INDETERMINANT_DEFAULTS["force_suffix"] = False
