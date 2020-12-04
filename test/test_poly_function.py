"""Testing functions used for numpoly only functionality."""
from pytest import raises

import sympy
import numpy
import numpoly

X, Y = numpoly.symbols("q0"), numpoly.symbols("q1")
POLY1 = numpoly.polynomial([[1, X, X-1, X**2],
                            [Y, Y-1, Y**2, 1],
                            [X-1, X**2, 1, X],
                            [Y**2, 1, Y, Y-1]])


def test_call():
    poly = X+Y
    with raises(TypeError):
        poly(1, q0=2)
    with raises(TypeError):
        poly(1, 2, q1=3)
    with raises(TypeError):
        poly(not_an_arg=45)
    poly = numpoly.polynomial([2, X-Y+1])
    assert numpy.all(poly(q0=Y) == [2, 1])


def test_ndpoly():
    poly = numpoly.ndpoly(exponents=[(1,)], shape=(), names="q0")
    poly["<"] = 1
    assert poly == X
    poly = numpoly.ndpoly(exponents=[(1,)], shape=(), names=X)
    poly["<"] = 1
    assert poly == X
    poly = numpoly.ndpoly(
        exponents=[(1, 0), (0, 1)], shape=(), names=("q0" ,"q1"))
    poly["<;"] = 2
    poly[";<"] = 3
    assert poly == 2*X+3*Y
    with numpoly.global_options(varname_filter=r"Q\d+"):
        poly = numpoly.ndpoly(
            exponents=[(1, 0), (0, 1)], shape=(2,), names="Q:2")
        poly["<;"] = [1, 0]
        poly[";<"] = [0, 1]
        assert numpy.all(poly == numpoly.symbols("Q0 Q1"))


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

    x1 = numpoly.polynomial(-1.715408531156317e+18)
    x2 = numpoly.polynomial(3.097893826691672e+18)
    divided, remainder = numpoly.poly_divmod(x1, x2)
    assert numpoly.allclose(x1, x2*divided)
    assert remainder == 0

    # No candidate and no infinity loops when cutoff is reached:
    assert numpoly.poly_function.get_division_candidate(x1, x2, cutoff=1) is None

    x1 = (24*3600)**2*398600.4415
    x2 = 2.689498059130061e-63*X**2+4.351432444850692e-27*X+1760083471.506941
    divided, remainder = numpoly.poly_divmod(x1, x2)
    assert numpoly.allclose(x1, x2*divided+remainder)
    assert numpy.allclose(remainder, x1)

    assert divmod(X+3, 2) == (0.5*X+1.5, 0)
    assert divmod(Y+3, 2) == (0.5*Y+1.5, 0)
    assert divmod(3, numpoly.polynomial(2)) == (1.5, 0)
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


def test_poly_remainder():
    poly = numpoly.polynomial([[1, 2*X], [3*Y+X, 4]])
    assert numpy.all(numpoly.poly_remainder(poly, 2) == [[0, 0], [0, 0]])
    assert numpy.all(numpy.array([1, 2]) % poly == [[0, 2], [1, 0]])
    assert numpy.all(poly % numpy.array([1, 2]) == 0)
    assert numpy.all(poly.__mod__(numpy.array([1, 2])) == 0)
    assert numpy.all(poly.__rmod__(numpy.array([1, 2])) == [[0, 2], [1, 0]])


def test_sortable_proxy():
    assert numpy.all(numpoly.sortable_proxy(numpy.arange(4)) == [0, 1, 2, 3])
    assert numpy.all(numpoly.sortable_proxy(-numpy.arange(4)) == [3, 2, 1, 0])

    assert numpy.all(numpoly.sortable_proxy(numpy.arange(4)*X) == [0, 1, 2, 3])
    assert numpy.all(numpoly.sortable_proxy(-numpy.arange(4)*X) == [0, 3, 2, 1])
    assert numpy.all(numpoly.sortable_proxy(X**numpy.arange(4)) == [0, 1, 2, 3])
    assert numpy.all(numpoly.sortable_proxy(-X**numpy.arange(4)) == [0, 1, 2, 3])
    assert numpy.all(numpoly.sortable_proxy(X**4+X**numpy.arange(4)) == [0, 1, 2, 3])
    assert numpy.all(numpoly.sortable_proxy(X**4-X**numpy.arange(4)) == [0, 1, 2, 3])

    poly = numpoly.polynomial([1, X, Y, X**2, X*Y, Y**2])

    proxy = numpoly.sortable_proxy(poly, graded=False, reverse=False)
    assert numpy.all(poly[numpy.argsort(proxy)] == [1, X, X**2, Y, X*Y, Y**2])
    proxy = numpoly.sortable_proxy(poly, graded=True, reverse=False)
    assert numpy.all(poly[numpy.argsort(proxy)] == [1, X, Y, X**2, X*Y, Y**2])
    proxy = numpoly.sortable_proxy(poly, graded=True, reverse=True)
    assert numpy.all(poly[numpy.argsort(proxy)] == [1, Y, X, Y**2, X*Y, X**2])
    proxy = numpoly.sortable_proxy(poly, graded=False, reverse=True)
    assert numpy.all(poly[numpy.argsort(proxy)] == [1, Y, Y**2, X, X*Y, X**2])

    assert numpy.all(numpoly.sortable_proxy(POLY1) ==
                     [[ 0,  4,  5,  8],
                      [10, 11, 14,  1],
                      [ 6,  9,  2,  7],
                      [15,  3, 12, 13]])
    assert numpy.all(numpoly.sortable_proxy(POLY1, graded=True) ==
                     [[ 0,  4,  5, 12],
                      [ 8,  9, 14,  1],
                      [ 6, 13,  2,  7],
                      [15,  3, 10, 11]])
    assert numpy.all(numpoly.sortable_proxy(POLY1, reverse=True) ==
                     [[ 0, 10, 11, 14],
                      [ 4,  5,  8,  1],
                      [12, 15,  2, 13],
                      [ 9,  3,  6,  7]])


def test_lead_exponent():
    empty = numpoly.lead_exponent([])
    assert empty.size == 0
    assert empty.dtype == int
    assert empty.shape == (0, 1)
    exponents = numpoly.lead_exponent(POLY1)
    xs, ys = exponents[:, :, 0], exponents[:, :, 1]
    assert numpy.all(xs == [[0, 1, 1, 2], [0, 0, 0, 0],
                            [1, 2, 0, 1], [0, 0, 0, 0]])
    assert numpy.all(ys == [[0, 0, 0, 0], [1, 1, 2, 0],
                            [0, 0, 0, 0], [2, 0, 1, 1]])


def test_lead_coefficient():
   empty = numpoly.lead_coefficient([])
   assert empty.size == 0
   assert empty.dtype == int
   assert empty.shape == (0,)


def test_set_dimensions():
    """Tests for numpoly.set_dimensions."""
    assert numpoly.set_dimensions(POLY1).names == ("q0", "q1", "q2")
    assert numpoly.set_dimensions(POLY1, 1).names == ("q0",)
    assert numpoly.set_dimensions(POLY1, 2).names == ("q0", "q1")
    assert numpoly.set_dimensions(POLY1, 3).names == ("q0", "q1", "q2")
    assert numpoly.set_dimensions(POLY1, 4).names == ("q0", "q1", "q2", "q3")
