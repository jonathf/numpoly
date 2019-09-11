"""Testing functions used for numpy compatible."""
from packaging.version import parse
import numpy
from numpoly import polynomial
import numpoly

if parse(numpy.__version__) < parse("1.17.0"):
    # use the __array_function__ interface (Python 3 in practice)
    INTERFACE = numpoly
else:
    # Use internal interface (Python 2 in practice)
    INTERFACE = numpy

X, Y = numpoly.symbols("X Y")


def test_numpy_absolute():
    assert abs(-X) == abs(X) == X
    assert numpy.all(abs(polynomial([X-Y, Y-4])) == polynomial([X+Y, Y+4]))


def test_numpy_add():
    assert X+3 == 3+X
    assert numpy.all(4 + polynomial([1, X, Y]) == polynomial([5, 4+X, 4+Y]))
    assert numpy.all(polynomial([0, X]) + polynomial([Y, 0]) == [Y, X])
    assert numpy.all(polynomial([[1, X], [Y, X*Y]]) + [2, X] ==
                     polynomial([[3, 2*X], [2+Y, X+X*Y]]))


def test_numpy_any():
    poly = polynomial([[0, Y], [0, 0]])
    assert INTERFACE.any(poly)
    assert numpy.all(INTERFACE.any(poly, axis=0) == [False, True])
    assert numpy.all(INTERFACE.any(poly, axis=-1, keepdims=True) ==
                     [[True], [False]])


def test_numpy_all():
    poly = polynomial([[0, Y], [X, 1]])
    assert not INTERFACE.all(poly)
    assert numpy.all(INTERFACE.all(poly, axis=0) == [False, True])
    assert numpy.all(INTERFACE.all(poly, axis=-1, keepdims=True) ==
                     [[False], [True]])


def test_numpy_around():
    poly = 123.45*X+Y
    assert INTERFACE.around(poly) == 123.*X+Y
    assert INTERFACE.around(poly, decimals=1) == 123.4*X+Y
    assert INTERFACE.around(poly, decimals=-2) == 100.*X


def test_numpy_array_repr():
    assert repr(4+6*X**2) == "polynomial(4+6*X**2)"
    assert (repr(polynomial([1., -5*X, 3-X**2])) ==
            "polynomial([1.0, -5.0*X, 3.0-X**2])")
    assert repr(polynomial([[[1, 2], [5, Y]]])) == """\
polynomial([[[1, 2],
             [5, Y]]])"""


def test_numpy_array_str():
    assert str(4+6*X**2) == "4+6*X**2"
    assert str(polynomial([1., -5*X, 3-X**2])) == "[1.0 -5.0*X 3.0-X**2]"
    assert str(polynomial([[[1, 2], [5, Y]]])) == """\
[[[1 2]
  [5 Y]]]"""


def test_numpy_common_type():
    assert INTERFACE.common_type(numpy.array(2, dtype=numpy.float32)) == numpy.float32
    assert INTERFACE.common_type(numpoly.symbols("x")) == numpy.float64
    assert INTERFACE.common_type(numpy.arange(3), 1j*numpoly.symbols("x"), 45) == numpy.complex128


def test_numpy_concatenate():
    poly1 = polynomial([[0, Y], [X, 1]])
    assert numpy.all(INTERFACE.concatenate([poly1, poly1]) ==
                     polynomial([[0, Y], [X, 1], [0, Y], [X, 1]]))
    assert numpy.all(INTERFACE.concatenate([poly1, [[X*Y, 1]]], 0) ==
                     polynomial([[0, Y], [X, 1], [X*Y, 1]]))
    assert numpy.all(INTERFACE.concatenate([poly1, [[X*Y], [1]]], 1) ==
                     polynomial([[0, Y, X*Y], [X, 1, 1]]))
    assert numpy.all(INTERFACE.concatenate([poly1, poly1], 1) ==
                     polynomial([[0, Y, 0, Y], [X, 1, X, 1]]))


def test_numpy_cumsum():
    poly1 = polynomial([[0, Y], [X, 1]])
    assert numpy.all(INTERFACE.cumsum(poly1) == [0, Y, X+Y, 1+X+Y])
    assert numpy.all(INTERFACE.cumsum(poly1, axis=0) == [[0, Y], [X, Y+1]])
    assert numpy.all(INTERFACE.cumsum(poly1, axis=1) == [[0, Y], [X, X+1]])

def test_numpy_floor_divide():
    poly = polynomial([[0., 2.*Y], [X, 2.]])
    assert numpy.all(poly // 2 == polynomial([[0, Y], [0, 1]]))
    assert numpy.all(poly // [1, 2] == polynomial([[0, Y], [X, 1]]))
    assert numpy.all(poly // [[1, 2], [2, 1]] == polynomial([[0, Y], [0, 2]]))


def test_numpy_inner():
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert INTERFACE.inner(poly1, poly2) == Y


def test_numpy_logical_or():
    assert numpy.all(numpy.logical_or(0, [1, X]) == [1, X])
    assert numpy.all(numpy.logical_or(1, [0, X]) == [1, 1])
    assert numpy.all(numpy.logical_or([0, Y], [0, X]) == [0, Y])


def test_numpy_multiply():
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert numpy.all(2*poly == [[0, 4+2*Y], [2*X, 4]])
    assert numpy.all([X, 1]*poly == [[0, 2+Y], [X*X, 2]])
    assert numpy.all([[X, 1], [Y, 0]]*poly == [[0, 2+Y], [X*Y, 0]])


def test_numpy_negative():
    poly = polynomial([[X, -Y], [-4, Y]])
    assert -(X-Y-1) == 1-X+Y
    assert numpy.all(-poly == [[-X, Y], [4, -Y]])


def test_numpy_not_equal():
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert numpy.all(([X, 2+Y] != poly) == [[True, False], [False, True]])
    assert numpy.all((X != poly) == [[True, True], [False, True]])


def test_numpy_true_divide():
    poly = polynomial([[0, Y], [X, 1]])
    assert numpy.all(poly / 2 == polynomial([[0, 0.5*Y], [0.5*X, 0.5]]))
    assert numpy.all(poly / [1, 2] == polynomial([[0, 0.5*Y], [X, 0.5]]))
    assert numpy.all(poly / [[1, 2], [2, 1]] ==
                     polynomial([[0, 0.5*Y], [0.5*X, 1]]))

def test_numpy_outer():
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert numpy.all(INTERFACE.outer(poly1, poly2) == [[0, 0], [X*Y+Y, Y]])


def test_numpy_positive():
    poly = polynomial([[0, Y], [X, 1]])
    assert numpy.all(poly == +poly)
    assert poly is not +poly


def test_numpy_power():
    poly = polynomial([[0, Y], [X-1, 2]])
    assert numpy.all(X**[2] == polynomial([X**2]))
    assert numpy.all(polynomial([X])**[2] == polynomial([X**2]))
    assert numpy.all(polynomial([X, Y])**[2] == polynomial([X**2, Y**2]))
    assert numpy.all(polynomial([X])**[1, 2] == polynomial([X, X**2]))
    assert numpy.all((X*Y)**[0, 1, 2, 3] == [1, X*Y, X**2*Y**2, X**3*Y**3])
    assert numpy.all(poly ** 2 == polynomial([[0, Y**2], [X*X-2*X+1, 4]]))
    assert numpy.all(poly ** [1, 2] == polynomial([[0, Y**2], [X-1, 4]]))
    assert numpy.all(poly ** [[1, 2], [2, 1]] ==
                     polynomial([[0, Y**2], [X*X-2*X+1, 2]]))


def test_numpy_subtract():
    assert -X+3 == 3-X
    assert numpy.all(4 - polynomial([1, X, Y]) == polynomial([3, 4-X, 4-Y]))
    assert numpy.all(polynomial([0, X]) - polynomial([Y, 0]) == [-Y, X])
    assert numpy.all(polynomial([[1, X], [Y, X*Y]]) - [2, X] ==
                     polynomial([[-1, 0], [Y-2, X*Y-X]]))


def test_numpy_sum():
    poly = polynomial([[1, 5*X], [X+3, -Y]])
    assert INTERFACE.sum(poly) == -Y+X*6+4
    assert numpy.all(INTERFACE.sum(poly, axis=0) == polynomial([X+4, -Y+X*5]))
    assert numpy.all(INTERFACE.sum(poly, axis=-1, keepdims=True) ==
                     polynomial([[X*5+1], [X-Y+3]]))
