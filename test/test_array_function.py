"""Testing functions used for numpy compatible."""
from packaging.version import parse
import pytest
import numpy
from numpoly import polynomial
import numpoly

if parse(numpy.__version__) < parse("1.17.0"):
    # Use internal interface only (Python 2 in practice)
    INTERFACES = [numpoly]
else:
    # use both internal and __array_function__ interface (Python 3 in practice)
    INTERFACES = [numpoly, numpy]


@pytest.fixture(params=INTERFACES)
def interface(request):
    return request.param


X, Y = numpoly.symbols("X Y")


def test_numpy_absolute(interface):
    assert abs(-X) == abs(X) == X
    assert numpy.all(abs(polynomial([X-Y, Y-4])) == polynomial([X+Y, Y+4]))
    assert numpy.all(interface.absolute(polynomial([X-Y, Y-4])) == polynomial([X+Y, Y+4]))


def test_numpy_add(interface):
    assert X+3 == 3+X
    assert numpy.all(4 + polynomial([1, X, Y]) == [5, 4+X, 4+Y])
    assert numpy.all(interface.add(4, polynomial([1, X, Y])) == [5, 4+X, 4+Y])
    assert numpy.all(polynomial([0, X]) + polynomial([Y, 0]) == [Y, X])
    assert numpy.all(interface.add(polynomial([0, X]), polynomial([Y, 0])) == [Y, X])
    assert numpy.all(polynomial([[1, X], [Y, X*Y]]) + [2, X] ==
                     polynomial([[3, 2*X], [2+Y, X+X*Y]]))
    assert numpy.all(interface.add(polynomial([[1, X], [Y, X*Y]]), [2, X]) ==
                     polynomial([[3, 2*X], [2+Y, X+X*Y]]))


def test_numpy_any(interface):
    poly = polynomial([[0, Y], [0, 0]])
    assert interface.any(poly)
    assert poly.any()
    assert numpy.all(interface.any(poly, axis=0) == [False, True])
    assert numpy.all(poly.any(axis=0) == [False, True])
    assert numpy.all(interface.any(poly, axis=-1, keepdims=True) == [[True], [False]])
    assert numpy.all(poly.any(axis=-1, keepdims=True) == [[True], [False]])


def test_numpy_all(interface):
    poly = polynomial([[0, Y], [X, 1]])
    assert not interface.all(poly)
    assert not poly.all()
    assert numpy.all(interface.all(poly, axis=0) == [False, True])
    assert numpy.all(poly.all(axis=0) == [False, True])
    assert numpy.all(interface.all(poly, axis=-1, keepdims=True) == [[False], [True]])
    assert numpy.all(poly.all(axis=-1, keepdims=True) == [[False], [True]])


def test_numpy_allclose(interface):
    poly1 = numpoly.polynomial([1e10*X, 1e-7])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-8])
    assert not interface.allclose(poly1, poly2)
    poly1 = numpoly.polynomial([1e10*X, 1e-8])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-9])
    assert interface.allclose(poly1, poly2)
    poly2 = numpoly.polynomial([1e10*Y, 1e-8])
    assert not interface.allclose(poly1, poly2)


def test_numpy_around(interface):
    poly = 123.45*X+Y
    assert interface.around(poly) == 123.*X+Y
    assert poly.round() == 123.*X+Y
    assert interface.around(poly, decimals=1) == 123.4*X+Y
    assert poly.round(decimals=1) == 123.4*X+Y
    assert interface.around(poly, decimals=-2) == 100.*X
    assert poly.round(decimals=-2) == 100.*X


def test_numpy_array_repr(interface):
    assert repr(4+6*X**2) == "polynomial(4+6*X**2)"
    assert interface.array_repr(4+6*X**2) == "polynomial(4+6*X**2)"
    assert (repr(polynomial([1., -5*X, 3-X**2])) ==
            "polynomial([1.0, -5.0*X, 3.0-X**2])")
    assert (interface.array_repr(polynomial([1., -5*X, 3-X**2])) ==
            "polynomial([1.0, -5.0*X, 3.0-X**2])")
    assert repr(polynomial([[[1, 2], [5, Y]]])) == """\
polynomial([[[1, 2],
             [5, Y]]])"""
    assert interface.array_repr(polynomial([[[1, 2], [5, Y]]])) == """\
polynomial([[[1, 2],
             [5, Y]]])"""


def test_numpy_array_str(interface):
    assert str(4+6*X**2) == "4+6*X**2"
    assert interface.array_str(4+6*X**2) == "4+6*X**2"
    assert str(polynomial([1., -5*X, 3-X**2])) == "[1.0 -5.0*X 3.0-X**2]"
    assert interface.array_str(polynomial([1., -5*X, 3-X**2])) == "[1.0 -5.0*X 3.0-X**2]"
    assert str(polynomial([[[1, 2], [5, Y]]])) == """\
[[[1 2]
  [5 Y]]]"""
    assert interface.array_str(polynomial([[[1, 2], [5, Y]]])) == """\
[[[1 2]
  [5 Y]]]"""


def test_numpy_common_type(interface):
    assert interface.common_type(numpy.array(2, dtype=numpy.float32)) == numpy.float32
    assert interface.common_type(X) == numpy.float64
    assert interface.common_type(numpy.arange(3), 1j*X, 45) == numpy.complex128


def test_numpy_concatenate(interface):
    poly1 = polynomial([[0, Y], [X, 1]])
    assert numpy.all(interface.concatenate([poly1, poly1]) ==
                     polynomial([[0, Y], [X, 1], [0, Y], [X, 1]]))
    assert numpy.all(interface.concatenate([poly1, [[X*Y, 1]]], 0) ==
                     polynomial([[0, Y], [X, 1], [X*Y, 1]]))
    assert numpy.all(interface.concatenate([poly1, [[X*Y], [1]]], 1) ==
                     polynomial([[0, Y, X*Y], [X, 1, 1]]))
    assert numpy.all(interface.concatenate([poly1, poly1], 1) ==
                     polynomial([[0, Y, 0, Y], [X, 1, X, 1]]))


def test_numpy_cumsum(interface):
    poly1 = polynomial([[0, Y], [X, 1]])
    assert numpy.all(interface.cumsum(poly1) == [0, Y, X+Y, 1+X+Y])
    assert numpy.all(poly1.cumsum() == [0, Y, X+Y, 1+X+Y])
    assert numpy.all(interface.cumsum(poly1, axis=0) == [[0, Y], [X, Y+1]])
    assert numpy.all(poly1.cumsum( axis=0) == [[0, Y], [X, Y+1]])
    assert numpy.all(interface.cumsum(poly1, axis=1) == [[0, Y], [X, X+1]])
    assert numpy.all(poly1.cumsum(axis=1) == [[0, Y], [X, X+1]])


def test_numpy_floor_divide(interface):
    poly = polynomial([[0., 2.*Y], [X, 2.]])
    assert numpy.all(poly // 2 == polynomial([[0, Y], [0, 1]]))
    assert numpy.all(interface.floor_divide(poly, 2) == polynomial([[0, Y], [0, 1]]))
    assert numpy.all(poly // [1, 2] == polynomial([[0, Y], [X, 1]]))
    assert numpy.all(interface.floor_divide(poly, [1, 2]) == polynomial([[0, Y], [X, 1]]))
    assert numpy.all(poly // [[1, 2], [2, 1]] == polynomial([[0, Y], [0, 2]]))
    assert numpy.all(interface.floor_divide(poly, [[1, 2], [2, 1]]) == polynomial([[0, Y], [0, 2]]))


def test_numpy_inner(interface):
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert interface.inner(poly1, poly2) == Y


def test_numpy_isclose(interface):
    poly1 = numpoly.polynomial([1e10*X, 1e-7])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-8])
    assert numpy.all(interface.isclose(poly1, poly2) == [True, False])
    poly1 = numpoly.polynomial([1e10*X, 1e-8])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-9])
    assert numpy.all(interface.isclose(poly1, poly2) == [True, True])
    poly2 = numpoly.polynomial([1e10*Y, 1e-8])
    assert numpy.all(interface.isclose(poly1, poly2) == [False, True])


def test_numpy_isfinite(interface):
    assert interface.isfinite(X)
    assert not interface.isfinite(numpy.nan*X)
    poly = numpoly.polynomial([numpy.log(-1.), X, numpy.log(0)])
    assert numpy.all(interface.isfinite(poly) == [False, True, False])


def test_numpy_logical_and(interface):
    poly1 = numpoly.polynomial([0, X])
    poly2 = numpoly.polynomial([1, X])
    poly3 = numpoly.polynomial([0, Y])
    assert numpy.all(interface.logical_and(1, poly1) == [False, True])
    assert numpy.all(interface.logical_and(1, poly2) == [True, True])
    assert numpy.all(interface.logical_and(poly2, poly3) == [False, True])


def test_numpy_logical_or(interface):
    poly1 = numpoly.polynomial([0, X])
    poly2 = numpoly.polynomial([1, X])
    poly3 = numpoly.polynomial([0, Y])
    assert numpy.all(interface.logical_or(1, poly1) == [True, True])
    assert numpy.all(interface.logical_or(0, poly1) == [False, True])
    assert numpy.all(interface.logical_or(poly2, poly3) == [True, True])


def test_numpy_multiply(interface):
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert numpy.all(2*poly == [[0, 4+2*Y], [2*X, 4]])
    assert numpy.all(interface.multiply(2, poly) == [[0, 4+2*Y], [2*X, 4]])
    assert numpy.all([X, 1]*poly == [[0, 2+Y], [X*X, 2]])
    assert numpy.all(interface.multiply([X, 1], poly) == [[0, 2+Y], [X*X, 2]])
    assert numpy.all([[X, 1], [Y, 0]]*poly == [[0, 2+Y], [X*Y, 0]])
    assert numpy.all(interface.multiply([[X, 1], [Y, 0]], poly) == [[0, 2+Y], [X*Y, 0]])


def test_numpy_mean(interface):
    poly = numpoly.polynomial([[1, 2*X], [3*Y+X, 4]])
    assert interface.mean(poly) == 1.25+0.75*Y+0.75*X
    assert poly.mean() == 1.25+0.75*Y+0.75*X
    assert numpy.all(interface.mean(poly, axis=0) == [0.5+1.5*Y+0.5*X, 2.0+X])
    assert numpy.all(poly.mean(axis=0) == [0.5+1.5*Y+0.5*X, 2.0+X])
    assert numpy.all(interface.mean(poly, axis=1) == [0.5+X, 2.0+1.5*Y+0.5*X])
    assert numpy.all(poly.mean(axis=1) == [0.5+X, 2.0+1.5*Y+0.5*X])

def test_numpy_negative(interface):
    poly = polynomial([[X, -Y], [-4, Y]])
    assert -(X-Y-1) == 1-X+Y
    assert interface.negative(X-Y-1) == 1-X+Y
    assert numpy.all(interface.negative(poly) == [[-X, Y], [4, -Y]])


def test_numpy_not_equal(interface):
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert numpy.all(([X, 2+Y] != poly) == [[True, False], [False, True]])
    assert numpy.all(interface.not_equal([X, 2+Y], poly) ==
                     [[True, False], [False, True]])
    assert numpy.all((X != poly) == [[True, True], [False, True]])
    assert numpy.all(interface.not_equal(X, poly) == [[True, True], [False, True]])


def test_numpy_true_divide(interface):
    poly = polynomial([[0, Y], [X, 1]])
    assert numpy.all(poly / 2 == polynomial([[0, 0.5*Y], [0.5*X, 0.5]]))
    assert numpy.all(interface.divide(poly, 2) ==
                     polynomial([[0, 0.5*Y], [0.5*X, 0.5]]))
    assert numpy.all(poly / [1, 2] == polynomial([[0, 0.5*Y], [X, 0.5]]))
    assert numpy.all(interface.divide(poly, [1, 2]) ==
                     polynomial([[0, 0.5*Y], [X, 0.5]]))
    assert numpy.all(poly / [[1, 2], [2, 1]] ==
                     polynomial([[0, 0.5*Y], [0.5*X, 1]]))
    assert numpy.all(interface.divide(poly, [[1, 2], [2, 1]]) ==
                     polynomial([[0, 0.5*Y], [0.5*X, 1]]))

def test_numpy_outer(interface):
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert numpy.all(interface.outer(poly1, poly2) == [[0, 0], [X*Y+Y, Y]])


def test_numpy_positive(interface):
    poly = polynomial([[0, Y], [X, 1]])
    assert numpy.all(poly == +poly)
    assert numpy.all(poly == interface.positive(poly))
    assert poly is not +poly
    assert poly is not interface.positive(poly)


def test_numpy_power(interface):
    poly = polynomial([[0, Y], [X-1, 2]])
    assert numpy.all(X**[2] == polynomial([X**2]))
    assert numpy.all(interface.power(X, [2]) == polynomial([X**2]))
    assert numpy.all(polynomial([X])**[2] == polynomial([X**2]))
    assert numpy.all(interface.power(polynomial([X]), [2]) == polynomial([X**2]))
    assert numpy.all(polynomial([X, Y])**[2] == polynomial([X**2, Y**2]))
    assert numpy.all(interface.power(polynomial([X, Y]), [2]) == polynomial([X**2, Y**2]))
    assert numpy.all(polynomial([X])**[1, 2] == polynomial([X, X**2]))
    assert numpy.all(interface.power(polynomial([X]), [1, 2]) == polynomial([X, X**2]))
    assert numpy.all((X*Y)**[0, 1, 2, 3] == [1, X*Y, X**2*Y**2, X**3*Y**3])
    assert numpy.all(interface.power(X*Y, [0, 1, 2, 3]) == [1, X*Y, X**2*Y**2, X**3*Y**3])
    assert numpy.all(poly ** 2 == polynomial([[0, Y**2], [X*X-2*X+1, 4]]))
    assert numpy.all(interface.power(poly, 2) == polynomial([[0, Y**2], [X*X-2*X+1, 4]]))
    assert numpy.all(poly ** [1, 2] == polynomial([[0, Y**2], [X-1, 4]]))
    assert numpy.all(interface.power(poly, [1, 2]) == polynomial([[0, Y**2], [X-1, 4]]))
    assert numpy.all(poly ** [[1, 2], [2, 1]] ==
                     polynomial([[0, Y**2], [X*X-2*X+1, 2]]))
    assert numpy.all(interface.power(poly, [[1, 2], [2, 1]]) ==
                     polynomial([[0, Y**2], [X*X-2*X+1, 2]]))


def test_numpy_rint(interface):
    poly = numpoly.polynomial([-1.7*X, X-1.5])
    assert numpy.all(interface.rint(poly) == [-2*X, X-2])


def test_numpy_square(interface):
    assert interface.square(X+Y) == X**2+2*X*Y+Y**2
    assert (1+X)**2 == 1+2*X+X**2


def test_numpy_subtract(interface):
    assert -X+3 == 3-X
    assert numpy.all(4 - polynomial([1, X, Y]) == polynomial([3, 4-X, 4-Y]))
    assert numpy.all(interface.subtract(4, polynomial([1, X, Y])) == polynomial([3, 4-X, 4-Y]))
    assert numpy.all(polynomial([0, X]) - polynomial([Y, 0]) == [-Y, X])
    assert numpy.all(interface.subtract(polynomial([0, X]), polynomial([Y, 0])) == [-Y, X])
    assert numpy.all(polynomial([[1, X], [Y, X*Y]]) - [2, X] ==
                     polynomial([[-1, 0], [Y-2, X*Y-X]]))
    assert numpy.all(interface.subtract(polynomial([[1, X], [Y, X*Y]]), [2, X]) ==
                     polynomial([[-1, 0], [Y-2, X*Y-X]]))


def test_numpy_sum(interface):
    poly = polynomial([[1, 5*X], [X+3, -Y]])
    assert interface.sum(poly) == -Y+X*6+4
    assert poly.sum() == -Y+X*6+4
    assert numpy.all(interface.sum(poly, axis=0) == polynomial([X+4, -Y+X*5]))
    assert numpy.all(poly.sum(axis=0) == polynomial([X+4, -Y+X*5]))
    assert numpy.all(poly.sum(axis=-1, keepdims=True) ==
                     polynomial([[X*5+1], [X-Y+3]]))
