"""Testing functions used for numpy compatible."""
from __future__ import division
from pytest import raises
import numpy
from numpoly import polynomial
import numpoly

X, Y = numpoly.symbols("X Y")


def test_absolute(interface):
    assert abs(-X) == abs(X) == X
    assert numpy.all(interface.abs(polynomial([X-Y, Y-4])) == [X+Y, Y+4])


def test_add(interface):
    assert interface.add(X, 3) == 3+X
    assert numpy.all(interface.add(polynomial([1, X, Y]), 4) == [5, 4+X, 4+Y])
    assert numpy.all(interface.add(polynomial([0, X]), polynomial([Y, 0])) == [Y, X])
    assert numpy.all(interface.add(polynomial([[1, X], [Y, X*Y]]), [2, X]) == [[3, 2*X], [2+Y, X+X*Y]])


def test_any(interface):
    poly = polynomial([[0, Y], [0, 0]])
    assert interface.any(poly)
    assert numpy.all(interface.any(poly, axis=0) == [False, True])
    assert numpy.all(interface.any(poly, axis=-1, keepdims=True) == [[True], [False]])


def test_all(interface):
    poly = polynomial([[0, Y], [X, 1]])
    assert not interface.all(poly)
    assert numpy.all(interface.all(poly, axis=0) == [False, True])
    assert numpy.all(interface.all(poly, axis=-1, keepdims=True) == [[False], [True]])


def test_allclose(func_interface):
    poly1 = numpoly.polynomial([1e10*X, 1e-7])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-8])
    assert not func_interface.allclose(poly1, poly2)
    poly1 = numpoly.polynomial([1e10*X, 1e-8])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-9])
    assert func_interface.allclose(poly1, poly2)
    poly2 = numpoly.polynomial([1e10*Y, 1e-8])
    assert not func_interface.allclose(poly1, poly2)


def test_amax(func_interface):
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert func_interface.amax(poly) == X**4
    assert numpy.all(func_interface.amax(poly, axis=0) == [X**4, Y**2, X**2, Y**2])
    assert numpy.all(func_interface.amax(poly, axis=1) == [X**2, X**2, Y**2, X**4])
    assert numpy.all(func_interface.amax(poly.reshape(2, 2, 2, 2), axis=(0, 1)) ==
                     [[X**4, Y**2], [X**2, Y**2]])


def test_amin(func_interface):
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert func_interface.amin(poly) == 1
    assert numpy.all(func_interface.amin(poly, axis=0) == [1, 3, 2, X])
    assert numpy.all(func_interface.amin(poly, axis=1) == [1, 2, 3, Y])
    assert numpy.all(func_interface.amin(poly.reshape(2, 2, 2, 2), axis=(0, 1)) ==
                     [[1, 3], [2, X]])


def test_argmax(func_interface):
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert func_interface.argmax(poly) == 13
    assert numpy.all(func_interface.argmax(poly, axis=0) == [3, 3, 1, 0])
    assert numpy.all(func_interface.argmax(poly, axis=1) == [3, 2, 1, 1])


def test_argmin(func_interface):
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert func_interface.argmin(poly) == 0
    assert numpy.all(func_interface.argmin(poly, axis=0) == [0, 0, 2, 1])
    assert numpy.all(func_interface.argmin(poly, axis=1) == [0, 3, 2, 2])


def test_apply_along_axis(func_interface):
    np_array = numpy.arange(9).reshape(3, 3)
    assert numpy.all(func_interface.apply_along_axis(numpy.sum, 0, np_array) == [9, 12, 15])
    assert numpy.all(func_interface.apply_along_axis(numpy.sum, 1, np_array) == [3, 12, 21])
    poly1 = numpoly.polynomial([[X, X, X], [Y, Y, Y], [1, 2, 3]])
    assert numpy.all(func_interface.apply_along_axis(numpoly.sum, 0, poly1) == [X+Y+1, X+Y+2, X+Y+3])
    assert numpy.all(func_interface.apply_along_axis(numpoly.sum, 1, poly1) == [3*X, 3*Y, 6])


def test_apply_over_axes(func_interface):
    np_array = numpy.arange(9).reshape(3, 3)
    assert numpy.all(func_interface.apply_over_axes(numpy.sum, np_array, 0) == [9, 12, 15])
    assert numpy.all(func_interface.apply_over_axes(numpy.sum, np_array, 1) == [[3], [12], [21]])
    poly1 = numpoly.polynomial([[X, X, X], [Y, Y, Y], [1, 2, 3]])
    assert numpy.all(func_interface.apply_over_axes(numpoly.sum, poly1, 0) == [X+Y+1, X+Y+2, X+Y+3])
    assert numpy.all(func_interface.apply_over_axes(numpoly.sum, poly1, 1) == [[3*X], [3*Y], [6]])


def test_around(interface):
    poly = 123.45*X+Y
    assert interface.round(poly) == 123.*X+Y
    assert interface.round(poly, decimals=1) == 123.4*X+Y
    assert interface.round(poly, decimals=-2) == 100.*X


def test_array_repr(func_interface):
    assert repr(4+6*X**2) == "polynomial(4+6*X**2)"
    assert func_interface.array_repr(4+6*X**2) == "polynomial(4+6*X**2)"
    assert (repr(polynomial([1., -5*X, 3-X**2])) ==
            "polynomial([1.0, -5.0*X, 3.0-X**2])")
    assert (func_interface.array_repr(polynomial([1., -5*X, 3-X**2])) ==
            "polynomial([1.0, -5.0*X, 3.0-X**2])")
    assert repr(polynomial([[[1, 2], [5, Y]]])) == """\
polynomial([[[1, 2],
             [5, Y]]])"""
    assert func_interface.array_repr(polynomial([[[1, 2], [5, Y]]])) == """\
polynomial([[[1, 2],
             [5, Y]]])"""


def test_array_split(func_interface):
    test_split(func_interface)
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2 = func_interface.array_split(poly, 2, axis=1)
    assert numpy.all(part1 == [[[1, X], [X+Y, Y]]])
    assert numpy.all(part2 == [[X**2], [Y]])


def test_array_str(func_interface):
    assert str(4+6*X**2) == "4+6*X**2"
    assert func_interface.array_str(4+6*X**2) == "4+6*X**2"
    assert str(polynomial([1., -5*X, 3-X**2])) == "[1.0 -5.0*X 3.0-X**2]"
    assert func_interface.array_str(polynomial([1., -5*X, 3-X**2])) == "[1.0 -5.0*X 3.0-X**2]"
    assert str(polynomial([[[1, 2], [5, Y]]])) == """\
[[[1 2]
  [5 Y]]]"""
    assert func_interface.array_str(polynomial([[[1, 2], [5, Y]]])) == """\
[[[1 2]
  [5 Y]]]"""


def test_atleast_1d(func_interface):
    polys = [X, [X], [[X]], [[[X]]]]
    assert numpy.all(func_interface.atleast_1d(*polys) ==
                     [[X], [X], [[X]], [[[X]]]])


def test_atleast_2d(func_interface):
    polys = [X, [X], [[X]], [[[X]]]]
    assert numpy.all(func_interface.atleast_2d(*polys) ==
                     [[[X]], [[X]], [[X]], [[[X]]]])


def test_atleast_3d(func_interface):
    polys = [X, [X], [[X]], [[[X]]]]
    results = func_interface.atleast_3d(*polys)
    assert isinstance(results, list)
    assert len(results) == len(polys)
    assert all(result.shape == (1, 1, 1) for result in results)
    assert numpy.all(results == [[[[X]]], [[[X]]], [[[X]]], [[[X]]]])


def test_broadcast_array(func_interface):
    polys = [X, [[Y, 1]], [[X], [Y]], [[X, 1], [Y, 2]]]
    results = func_interface.broadcast_arrays(*polys)
    assert isinstance(results, list)
    assert len(results) == len(polys)
    assert all(result.shape == (2, 2) for result in results)
    assert numpy.all(results[0] == [[X, X], [X, X]])
    assert numpy.all(results[1] == [[Y, 1], [Y, 1]])
    assert numpy.all(results[2] == [[X, X], [Y, Y]])
    assert numpy.all(results[3] == [[X, 1], [Y, 2]])


def test_ceil(func_interface):
    poly = polynomial([-1.7*X, X-1.5, -0.2, 3.2+1.5*X, 1.7, 2.0])
    assert numpy.all(func_interface.ceil(poly) ==
                     [-X, -1.0+X, 0.0, 4.0+2.0*X, 2.0, 2.0])


def test_common_type(func_interface):
    assert func_interface.common_type(numpy.array(2, dtype=numpy.float32)) == numpy.float32
    assert func_interface.common_type(X) == numpy.float64
    assert func_interface.common_type(numpy.arange(3), 1j*X, 45) == numpy.complex128


def test_concatenate(func_interface):
    poly1 = polynomial([[0, Y], [X, 1]])
    assert numpy.all(func_interface.concatenate([poly1, poly1]) ==
                     [[0, Y], [X, 1], [0, Y], [X, 1]])
    assert numpy.all(func_interface.concatenate([poly1, [[X*Y, 1]]], 0) ==
                     [[0, Y], [X, 1], [X*Y, 1]])
    assert numpy.all(func_interface.concatenate([poly1, [[X*Y], [1]]], 1) ==
                     [[0, Y, X*Y], [X, 1, 1]])
    assert numpy.all(func_interface.concatenate([poly1, poly1], 1) ==
                     [[0, Y, 0, Y], [X, 1, X, 1]])

def test_count_nonzero(func_interface):
    poly1 = polynomial([[0, Y], [X, 1]])
    poly2 = polynomial([[0, Y, X, 0, 0], [3, 0, 0, 2, 19]])
    assert numpy.all(func_interface.count_nonzero(poly1) == 3)
    assert numpy.all(func_interface.count_nonzero(poly1, axis=0) == [1, 2])
    assert numpy.all(func_interface.count_nonzero(poly2, axis=0) == [1, 1, 1, 1, 1])
    assert numpy.all(func_interface.count_nonzero(X) == 1)


def test_cumsum(interface):
    poly1 = polynomial([[0, Y], [X, 1]])
    assert numpy.all(interface.cumsum(poly1) == [0, Y, X+Y, 1+X+Y])
    assert numpy.all(interface.cumsum(poly1, axis=0) == [[0, Y], [X, Y+1]])
    assert numpy.all(interface.cumsum(poly1, axis=1) == [[0, Y], [X, X+1]])


def test_divmod(func_interface):
    array = numpy.array([7, 11])
    quotient, remainder = func_interface.divmod(array, 5)
    assert numpy.all(quotient == [1, 2])
    assert numpy.all(remainder == [2, 1])
    with raises(numpoly.FeatureNotSupported):
        func_interface.divmod(array, X)
    with raises(numpoly.FeatureNotSupported):
        func_interface.divmod(X, X)


def test_dsplit(func_interface):
    poly = numpoly.polynomial([[[1, X], [X+Y, Y]]])
    part1, part2 = func_interface.dsplit(poly, 2)
    assert numpy.all(part1 == [[[1], [X+Y]]])
    assert numpy.all(part2 == [[[X], [Y]]])


def test_dstack(func_interface):
    poly1 = numpoly.polynomial([1, X, 2])
    poly2 = numpoly.polynomial([Y, 3, 4])
    assert numpy.all(func_interface.dstack([poly1, poly2]) ==
                     [[[1, Y], [X, 3], [2, 4]]])


def test_expand_dims(func_interface):
    poly1 = numpoly.polynomial([[1, X], [Y, 2], [3, 4]])
    assert func_interface.expand_dims(poly1, axis=0).shape == (1, 3, 2)
    assert func_interface.expand_dims(poly1, axis=1).shape == (3, 1, 2)
    assert func_interface.expand_dims(poly1, axis=2).shape == (3, 2, 1)
    array = numpy.arange(12).reshape(2, 3, 2)
    assert func_interface.expand_dims(array, axis=1).shape == (2, 1, 3, 2)


def test_floor(func_interface):
    poly = polynomial([-1.7*X, X-1.5, -0.2, 3.2+1.5*X, 1.7, 2.0])
    assert numpy.all(func_interface.floor(poly) ==
                     [-2.0*X, -2.0+X, -1.0, 3.0+X, 1.0, 2.0])


def test_floor_divide(interface):
    poly = polynomial([[0., 2.*Y], [X, 2.]])
    assert numpy.all(interface.floor_divide(poly, 2) == [[0, Y], [0, 1]])
    assert numpy.all(interface.floor_divide(poly, [1, 2]) == [[0, Y], [X, 1]])
    assert numpy.all(interface.floor_divide(poly, [[1, 2], [2, 1]]) == [[0, Y], [0, 2]])
    with raises(numpoly.FeatureNotSupported):
        interface.floor_divide(poly, X)
    with raises(numpoly.FeatureNotSupported):
        poly.__floordiv__(poly)
    with raises(numpoly.FeatureNotSupported):
        poly.__rfloordiv__(poly)

    out = numpoly.ndpoly(
        exponents=poly.exponents,
        shape=(2, 2),
        names=("X", "Y"),
        dtype=float,
    )
    numpoly.floor_divide(poly, 2, out=[out])
    assert numpy.all(out == [[0, Y], [0, 1]])


def test_hsplit(func_interface):
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2, part3 = func_interface.hsplit(poly, 3)
    assert numpy.all(part1 == [[1], [X+Y]])
    assert numpy.all(part2 == [[X], [Y]])
    assert numpy.all(part3 == [[X**2], [Y]])


def test_hstack(func_interface):
    poly1 = numpoly.polynomial([1, X, 2])
    poly2 = numpoly.polynomial([Y, 3, 4])
    assert numpy.all(func_interface.hstack([poly1, poly2]) ==
                     [1, X, 2, Y, 3, 4])


def test_inner(func_interface):
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert func_interface.inner(poly1, poly2) == Y


def test_isclose(func_interface):
    poly1 = numpoly.polynomial([1e10*X, 1e-7])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-8])
    assert numpy.all(func_interface.isclose(poly1, poly2) == [True, False])
    poly1 = numpoly.polynomial([1e10*X, 1e-8])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-9])
    assert numpy.all(func_interface.isclose(poly1, poly2) == [True, True])
    poly2 = numpoly.polynomial([1e10*Y, 1e-8])
    assert numpy.all(func_interface.isclose(poly1, poly2) == [False, True])


def test_isfinite(func_interface):
    assert func_interface.isfinite(X)
    assert not func_interface.isfinite(numpy.nan*X)
    poly = numpoly.polynomial([numpy.log(-1.), X, numpy.log(0)])
    assert numpy.all(func_interface.isfinite(poly) == [False, True, False])


def test_logical_and(func_interface):
    poly1 = numpoly.polynomial([0, X])
    poly2 = numpoly.polynomial([1, X])
    poly3 = numpoly.polynomial([0, Y])
    assert numpy.all(func_interface.logical_and(1, poly1) == [False, True])
    assert numpy.all(1 and poly1 == [0, X])
    assert numpy.all(func_interface.logical_and(1, poly2) == [True, True])
    assert numpy.all(1 and poly2 == [1, X])
    assert numpy.all(func_interface.logical_and(poly2, poly3) == [False, True])


def test_logical_or(func_interface):
    poly1 = numpoly.polynomial([0, X])
    poly2 = numpoly.polynomial([1, X])
    poly3 = numpoly.polynomial([0, Y])
    assert numpy.all(func_interface.logical_or(1, poly1) == [True, True])
    assert numpy.all(1 or poly1 == [1, 1])
    assert numpy.all(func_interface.logical_or(0, poly1) == [False, True])
    assert numpy.all(0 or poly1 == [0, X])
    assert numpy.all(func_interface.logical_or(poly2, poly3) == [True, True])


def test_matmul(func_interface):
    poly1 = numpoly.polynomial([[0, X], [1, Y]])
    poly2 = numpoly.polynomial([X, 2])
    assert numpy.all(func_interface.matmul(poly1, poly2) == [[X**2, 2*X], [X+X*Y, 2+2*Y]])
    assert func_interface.matmul(numpy.zeros((9, 5, 7, 4)), numpy.ones((9, 5, 4, 3))).shape == (9, 5, 7, 3)
    with raises(ValueError):
        func_interface.matmul(poly1, 4)
    with raises(ValueError):
        func_interface.matmul(3, poly2)


def test_mean(interface):
    poly = numpoly.polynomial([[1, 2*X], [3*Y+X, 4]])
    assert interface.mean(poly) == 1.25+0.75*Y+0.75*X
    assert numpy.all(interface.mean(poly, axis=0) == [0.5+1.5*Y+0.5*X, 2.0+X])
    assert numpy.all(interface.mean(poly, axis=1) == [0.5+X, 2.0+1.5*Y+0.5*X])


def test_max(interface):
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert interface.max(poly) == X**4
    assert numpy.all(interface.max(poly, axis=0) == [X**4, Y**2, X**2, Y**2])
    assert numpy.all(interface.max(poly, axis=1) == [X**2, X**2, Y**2, X**4])
    assert numpy.all(interface.max(poly.reshape(2, 2, 2, 2), axis=(0, 1)) ==
                     [[X**4, Y**2], [X**2, Y**2]])


def test_min(interface):
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert interface.min(poly) == 1
    assert numpy.all(interface.min(poly, axis=0) == [1, 3, 2, X])
    assert numpy.all(interface.min(poly, axis=1) == [1, 2, 3, Y])
    assert numpy.all(interface.min(poly.reshape(2, 2, 2, 2), axis=(0, 1)) ==
                     [[1, 3], [2, X]])


def test_moveaxis(func_interface):
    x = numpy.arange(6).reshape(1, 2, 3)
    assert numpy.all(func_interface.moveaxis(x, 0, -1) ==
                     [[[0], [1], [2]], [[3], [4], [5]]])
    assert numpy.all(func_interface.moveaxis(x, [0, 2], [2, 0]) ==
                     [[[0], [3]], [[1], [4]], [[2], [5]]])


def test_multiply(func_interface):
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert numpy.all(2*poly == [[0, 4+2*Y], [2*X, 4]])
    assert numpy.all(func_interface.multiply(2, poly) == [[0, 4+2*Y], [2*X, 4]])
    assert numpy.all([X, 1]*poly == [[0, 2+Y], [X*X, 2]])
    assert numpy.all(func_interface.multiply([X, 1], poly) == [[0, 2+Y], [X*X, 2]])
    assert numpy.all([[X, 1], [Y, 0]]*poly == [[0, 2+Y], [X*Y, 0]])
    assert numpy.all(func_interface.multiply([[X, 1], [Y, 0]], poly) == [[0, 2+Y], [X*Y, 0]])


def test_negative(func_interface):
    poly = polynomial([[X, -Y], [-4, Y]])
    assert -(X-Y-1) == 1-X+Y
    assert func_interface.negative(X-Y-1) == 1-X+Y
    assert numpy.all(func_interface.negative(poly) == [[-X, Y], [4, -Y]])

def test_nonzero(interface):
    poly = polynomial([[3*X, 0, 0], [0, 4*Y, 0], [5*X+Y, 6*X, 0]])
    assert numpy.all(poly[interface.nonzero(poly)] == [3*X, 4*Y, 5*X+Y, 6*X])
    assert numpy.all(interface.nonzero(X) == ([0],))


def test_not_equal(func_interface):
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert numpy.all(([X, 2+Y] != poly) == [[True, False], [False, True]])
    assert numpy.all(func_interface.not_equal([X, 2+Y], poly) ==
                     [[True, False], [False, True]])
    assert numpy.all((X != poly) == [[True, True], [False, True]])
    assert numpy.all(func_interface.not_equal(X, poly) == [[True, True], [False, True]])


def test_outer(func_interface):
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert numpy.all(func_interface.outer(poly1, poly2) == [[0, 0], [X*Y+Y, Y]])


def test_positive(func_interface):
    poly = polynomial([[0, Y], [X, 1]])
    assert numpy.all(poly == +poly)
    assert numpy.all(poly == func_interface.positive(poly))
    assert poly is not +poly
    assert poly is not func_interface.positive(poly)


def test_power(func_interface):
    poly = polynomial([[0, Y], [X-1, 2]])
    assert numpy.all(X**[2] == [X**2])
    assert numpy.all(func_interface.power(X, [2]) == [X**2])
    assert numpy.all(polynomial([X])**[2] == [X**2])
    assert numpy.all(func_interface.power(polynomial([X]), [2]) == [X**2])
    assert numpy.all(polynomial([X, Y])**[2] == [X**2, Y**2])
    assert numpy.all(func_interface.power(polynomial([X, Y]), [2]) == [X**2, Y**2])
    assert numpy.all(polynomial([X])**[1, 2] == [X, X**2])
    assert numpy.all(func_interface.power(polynomial([X]), [1, 2]) == [X, X**2])
    assert numpy.all((X*Y)**[0, 1, 2, 3] == [1, X*Y, X**2*Y**2, X**3*Y**3])
    assert numpy.all(func_interface.power(X*Y, [0, 1, 2, 3]) == [1, X*Y, X**2*Y**2, X**3*Y**3])
    assert numpy.all(poly ** 2 == [[0, Y**2], [X*X-2*X+1, 4]])
    assert numpy.all(func_interface.power(poly, 2) == [[0, Y**2], [X*X-2*X+1, 4]])
    assert numpy.all(poly ** [1, 2] == [[0, Y**2], [X-1, 4]])
    assert numpy.all(func_interface.power(poly, [1, 2]) == [[0, Y**2], [X-1, 4]])
    assert numpy.all(poly ** [[1, 2], [2, 1]] == [[0, Y**2], [X*X-2*X+1, 2]])
    assert numpy.all(func_interface.power(poly, [[1, 2], [2, 1]]) == [[0, Y**2], [X*X-2*X+1, 2]])


def test_prod(interface):
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    assert interface.prod(poly) == X**3*Y**3+X**4*Y**2
    assert numpy.all(interface.prod(poly, axis=0) == [Y+X, X*Y, X**2*Y])


def test_remainder(func_interface):
    assert numpy.all(func_interface.remainder([7, 11], 5) == [2, 1])
    with raises(numpoly.FeatureNotSupported):
        func_interface.remainder(X, X)
    with raises(numpoly.FeatureNotSupported):
        func_interface.remainder([1, 2], X)


def test_repeat(func_interface):
    poly = numpoly.polynomial([[1, X-1], [X**2, X]])
    assert numpy.all(func_interface.repeat(poly, 2) ==
                     [[1, -1+X], [1, -1+X], [X**2, X], [X**2, X]])
    assert numpy.all(func_interface.repeat(poly, 3, axis=1) ==
                     [[1, 1, 1, -1+X, -1+X, -1+X],
                      [X**2, X**2, X**2, X, X, X]])
    assert numpy.all(numpoly.repeat(poly, [1, 2], axis=0) ==
                     [[1, -1+X], [X**2, X], [X**2, X]])


def test_reshape(interface):
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    assert numpy.all(interface.reshape(poly, (3, 2)) ==
                     [[1, X], [X**2, X+Y], [Y, Y]])
    assert numpy.all(interface.reshape(poly, 6) ==
                     [1, X, X**2, X+Y, Y, Y])


def test_rint(func_interface):
    poly = numpoly.polynomial([-1.7*X, X-1.5])
    assert numpy.all(func_interface.rint(poly) == [-2*X, X-2])


def test_split(func_interface):
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2 = func_interface.split(poly, 2, axis=0)
    assert numpy.all(part1 == [1, X, X**2])
    assert numpy.all(part2 == [X+Y, Y, Y])
    part1, part2, part3 = func_interface.split(poly, 3, axis=1)
    assert numpy.all(part1 == [[1], [X+Y]])
    assert numpy.all(part2 == [[X], [Y]])
    assert numpy.all(part3 == [[X**2], [Y]])


def test_square(func_interface):
    assert func_interface.square(X+Y) == X**2+2*X*Y+Y**2
    assert (1+X)**2 == 1+2*X+X**2


def test_stack(func_interface):
    poly = polynomial([1, X, Y])
    assert numpy.all(func_interface.stack([poly, poly], axis=0) == [[1, X, Y], [1, X, Y]])
    assert numpy.all(func_interface.stack([poly, poly], axis=1) == [[1, 1], [X, X], [Y, Y]])


def test_subtract(func_interface):
    assert -X+3 == 3-X
    assert numpy.all(4 - polynomial([1, X, Y]) == [3, 4-X, 4-Y])
    assert numpy.all(func_interface.subtract(4, polynomial([1, X, Y])) == [3, 4-X, 4-Y])
    assert numpy.all(polynomial([0, X]) - polynomial([Y, 0]) == [-Y, X])
    assert numpy.all(func_interface.subtract(polynomial([0, X]), [Y, 0]) == [-Y, X])
    assert numpy.all(polynomial([[1, X], [Y, X*Y]]) - [2, X] ==
                     [[-1, 0], [Y-2, X*Y-X]])
    assert numpy.all(func_interface.subtract(polynomial([[1, X], [Y, X*Y]]), [2, X]) ==
                     [[-1, 0], [Y-2, X*Y-X]])


def test_sum(interface):
    poly = polynomial([[1, 5*X], [X+3, -Y]])
    assert interface.sum(poly) == -Y+X*6+4
    assert numpy.all(interface.sum(poly, axis=0) == [X+4, -Y+X*5])
    assert numpy.all(interface.sum(poly, axis=-1, keepdims=True) == [[X*5+1], [X-Y+3]])


def test_tranpose(func_interface):
    poly = numpoly.polynomial([[1, X-1], [X**2, X]])
    assert numpy.all(func_interface.transpose(poly) ==
                     [[1, X**2], [X-1, X]])
    assert numpy.all(poly.T == [[1, X**2], [X-1, X]])


def test_true_divide(func_interface):
    poly = polynomial([[0, Y], [X, 1]])
    assert numpy.all(func_interface.true_divide(poly, 2) == polynomial([[0, 0.5*Y], [0.5*X, 0.5]]))
    assert numpy.all(func_interface.true_divide(poly, [1, 2]) == [[0, 0.5*Y], [X, 0.5]])
    assert numpy.all(func_interface.true_divide(poly, [[1, 2], [2, 1]]) == [[0, 0.5*Y], [0.5*X, 1]])
    with raises(numpoly.FeatureNotSupported):
        func_interface.true_divide(poly, X)


def test_vsplit(func_interface):
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2 = func_interface.vsplit(poly, 2)
    assert numpy.all(part1 == [1, X, X**2])
    assert numpy.all(part2 == [X+Y, Y, Y])


def test_vstack(func_interface):
    poly1 = numpoly.polynomial([1, X, 2])
    poly2 = numpoly.polynomial([Y, 3, 4])
    assert numpy.all(func_interface.vstack([poly1, poly2]) ==
                     [[1, X, 2], [Y, 3, 4]])


def test_where(func_interface):
    poly1 = numpoly.polynomial([0, 4, 0])
    poly2 = numpoly.polynomial([Y, 0, X])
    assert numpy.all(func_interface.where([0, 1, 0], poly1, poly2) == [Y, 4, X])
    assert numpy.all(func_interface.where([1, 0, 1], poly1, poly2) == [0, 0, 0])
    assert func_interface.where(poly1) == (1,)
    assert numpy.all(func_interface.where(poly2)[0] == [0, 2])
