import numpy
from numpoly import polynomial, symbols

x, y = symbols("x y")


def test_numpy_absolute():
    assert abs(-x) == abs(x) == x
    assert numpy.all(abs(polynomial([x-y, y-4])) == polynomial([x+y, y+4]))


def test_numpy_add():
    assert x+3 == 3+x
    assert numpy.all(4 + polynomial([1, x, y]) == polynomial([5, 4+x, 4+y]))
    assert numpy.all(polynomial([0, x]) + polynomial([y, 0]) == [y, x])
    assert numpy.all(polynomial([[1, x], [y, x*y]]) + [2, x] ==
                     polynomial([[3, 2*x], [2+y, x+x*y]]))


def test_numpy_any():
    poly = polynomial([[0, y], [0, 0]])
    assert numpy.any(poly)
    assert numpy.all(numpy.any(poly, 0) == [False, True])
    assert numpy.all(numpy.any(poly, -1, keepdims=True) == [[True], [False]])


def test_numpy_all():
    poly = polynomial([[0, y], [x, 1]])
    assert not numpy.all(poly)
    assert numpy.all(numpy.all(poly, 0) == [False, True])
    assert numpy.all(numpy.all(poly, -1, keepdims=True) == [[False], [True]])


def test_numpy_concatenate():
    poly1 = polynomial([[0, y], [x, 1]])
    assert numpy.all(numpy.concatenate([poly1, poly1]) ==
                     polynomial([[0, y], [x, 1], [0, y], [x, 1]]))
    assert numpy.all(numpy.concatenate([poly1, [[x*y, 1]]], 0) ==
                     polynomial([[0, y], [x, 1], [x*y, 1]]))
    assert numpy.all(numpy.concatenate([poly1, [[x*y], [1]]], 1) ==
                     polynomial([[0, y, x*y], [x, 1, 1]]))
    assert numpy.all(numpy.concatenate([poly1, poly1], 1) ==
                     polynomial([[0, y, 0, y], [x, 1, x, 1]]))

def test_numpy_floor_divide():
    poly = polynomial([[0., 2.*y], [x, 2.]])
    assert numpy.all(poly // 2 == polynomial([[0, y], [0, 1]]))
    assert numpy.all(poly // [1, 2] == polynomial([[0, y], [x, 1]]))
    assert numpy.all(poly // [[1, 2], [2, 1]] ==
                     polynomial([[0, y], [0, 2]]))


def test_numpy_multiply():
    poly = polynomial([[0, 2+y], [x, 2]])
    assert numpy.all(2*poly == [[0, 4+2*y], [2*x, 4]])
    assert numpy.all([x, 1]*poly == [[0, 2+y], [x*x, 2]])
    assert numpy.all([[x, 1], [y, 0]]*poly == [[0, 2+y], [x*y, 0]])


def test_numpy_negative():
    poly = polynomial([[x, -y], [-4, y]])
    assert -(x-y-1) == 1-x+y
    assert numpy.all(-poly == [[-x, y], [4, -y]])


def test_numpy_not_equal():
    poly = polynomial([[0, 2+y], [x, 2]])
    assert numpy.all(([x, 2+y] != poly) == [[True, False], [False, True]])
    assert numpy.all((x != poly) == [[True, True], [False, True]])


def test_numpy_true_divide():
    poly = polynomial([[0, y], [x, 1]])
    assert numpy.all(poly / 2 == polynomial([[0, 0.5*y], [0.5*x, 0.5]]))
    assert numpy.all(poly / [1, 2] == polynomial([[0, 0.5*y], [x, 0.5]]))
    assert numpy.all(poly / [[1, 2], [2, 1]] ==
                     polynomial([[0, 0.5*y], [0.5*x, 1]]))

def test_numpy_positive():
    poly = polynomial([[0, y], [x, 1]])
    assert numpy.all(poly == +poly)
    assert poly is not +poly


def test_numpy_power():
    poly = polynomial([[0, y], [x-1, 2]])
    assert numpy.all(x**[2] == polynomial([x**2]))
    assert numpy.all(polynomial([x])**[2] == polynomial([x**2]))
    assert numpy.all(polynomial([x, y])**[2] == polynomial([x**2, y**2]))
    assert numpy.all(polynomial([x])**[1, 2] == polynomial([x, x**2]))
    assert numpy.all((x*y)**[0, 1, 2, 3] == [1, x*y, x**2*y**2, x**3*y**3])
    assert numpy.all(poly ** 2 == polynomial([[0, y**2], [x*x-2*x+1, 4]]))
    assert numpy.all(poly ** [1, 2] == polynomial([[0, y**2], [x-1, 4]]))
    assert numpy.all(poly ** [[1, 2], [2, 1]] ==
                     polynomial([[0, y**2], [x*x-2*x+1, 2]]))


def test_numpy_subtract():
    assert -x+3 == 3-x
    assert numpy.all(4 - polynomial([1, x, y]) == polynomial([3, 4-x, 4-y]))
    assert numpy.all(polynomial([0, x]) - polynomial([y, 0]) == [-y, x])
    assert numpy.all(polynomial([[1, x], [y, x*y]]) - [2, x] ==
                     polynomial([[-1, 0], [y-2, x*y-x]]))


def test_numpy_sum():
    poly = polynomial([[1, 5*x], [x+3, -y]])
    assert numpy.sum(poly) == -y+x*6+4
    assert numpy.all(numpy.sum(poly, axis=0) == polynomial([x+4, -y+x*5]))
    assert numpy.all(
        numpy.sum(poly, axis=-1, keepdims=True) == polynomial([[x*5+1], [x-y+3]]))


def test_numpy_array_str():
    assert str(4+6*x**2) == "4+6*x**2"
    assert str(polynomial([1., -5*x, 3-x**2])) == "[1.0 -5.0*x 3.0-x**2]"
    assert str(polynomial([[[1, 2], [5, y]]])) == """\
[[[1 2]
  [5 y]]]"""


def test_numpy_array_repr():
    assert repr(4+6*x**2) == "polynomial(4+6*x**2)"
    assert (repr(polynomial([1., -5*x, 3-x**2])) ==
            "polynomial([1.0, -5.0*x, 3.0-x**2])")
    assert repr(polynomial([[[1, 2], [5, y]]])) == """\
polynomial([[[1, 2],
             [5, y]]])"""
