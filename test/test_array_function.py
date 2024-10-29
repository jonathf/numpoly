"""Testing functions used for numpy compatible."""
from __future__ import division
from pytest import raises
import numpy
from numpoly import polynomial
import numpoly

X, Y = numpoly.variable(2)


def assert_equal(results, reference, c_contiguous=None,
                 f_contiguous=None, type_=None):
    """
    Assert that a return value for a function is the same as reference.

    Checks types, values, datatype, shape, C- and F-contiguous-ness.

    Args:
        results (numpy.ndarray, numpoly.ndpoly):
            The results to check are correct.
        reference (numpy.ndarray, numpoly.ndpoly):
            The reference the results is checked against. Input will be cast to
            the same type as `results`.
        c_contiguous (Optional[bool]):
            Check if `results` has correct `C_CONTIGUOUS` flag. Checked against
            `reference` if not provided.
        f_contiguous (Optional[bool]):
            Check if `results` has correct `F_CONTIGUOUS` flag. Checked against
            `reference` if not provided.
        type_ (Optional[type]):
            Check if `results` is correct type using `isinstance`. If not
            provided, results are checked to be legal numpy or numpoly type.

    """
    if type_ is None:
        assert isinstance(
            results, (bool, numpy.bool_, numpy.number, numpy.ndarray)), (
                f"unrecognized results type: {results}")
    else:
        assert isinstance(results, type_), (
            f"invalid results type: {results} != {type_}")
    if isinstance(results, numpoly.ndpoly):
        reference = numpoly.aspolynomial(reference)
    else:
        results = numpy.asarray(results)
        reference = numpy.asarray(reference)

    assert results.shape == reference.shape, (
        f"shape mismatch: {results} != {reference}")
    assert results.dtype == reference.dtype, (
        f"dtype mismatch: {results} != {reference}")

    if not isinstance(results, numpoly.ndpoly):
        assert numpy.allclose(results, reference), (
            f"value mismatch: {results} != {reference}")
    elif results.shape:
        assert numpy.all(results == reference), (
            f"value mismatch: {results} != {reference}")
    else:
        assert results == reference, (
            f"value mismatch: {results} != {reference}")

    if c_contiguous is None:
        c_contiguous = reference.flags["C_CONTIGUOUS"]
    assert results.flags["C_CONTIGUOUS"] == c_contiguous, (
        f"c_contiguous mismatch: {results} != {reference}")
    if f_contiguous is None:
        f_contiguous = reference.flags["F_CONTIGUOUS"]
    assert results.flags["F_CONTIGUOUS"] == f_contiguous, (
        f"f_contiguous mismatch: {results} != {reference}")


def test_absolute(interface):
    """Tests for numpoly.absolute."""
    assert_equal(X, abs(-X))
    assert_equal(abs(X), abs(-X))
    assert_equal(interface.abs(polynomial([X-Y, Y-4])), [X+Y, Y+4])


def test_add(interface):
    """Tests for numpoly.add."""
    assert_equal(interface.add(X, 3), 3+X)
    assert_equal(interface.add(polynomial([1, X, Y]), 4), [5, 4+X, 4+Y])
    assert_equal(interface.add(polynomial([0, X]), polynomial([Y, 0])), [Y, X])
    assert_equal(interface.add(polynomial(
        [[1, X], [Y, X*Y]]), [2, X]), [[3, 2*X], [2+Y, X+X*Y]])


def test_any(interface):
    """Tests for numpoly.any."""
    poly = polynomial([[0, Y], [0, 0]])
    assert_equal(interface.any(poly), True)
    assert_equal(interface.any(poly, axis=0), [False, True])
    assert_equal(
        interface.any(poly, axis=-1, keepdims=True), [[True], [False]])


def test_all(interface):
    """Tests for numpoly.all."""
    poly = polynomial([[0, Y], [X, 1]])
    assert_equal(interface.all(poly), False)
    assert_equal(interface.all(poly, axis=0), [False, True])
    assert_equal(
        interface.all(poly, axis=-1, keepdims=True), [[False], [True]])


def test_allclose(func_interface):
    """Tests for numpoly.allclose."""
    poly1 = numpoly.polynomial([1e10*X, 1e-7])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-8])
    assert_equal(func_interface.allclose(poly1, poly2), False, type_=bool)
    poly1 = numpoly.polynomial([1e10*X, 1e-8])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-9])
    assert_equal(func_interface.allclose(poly1, poly2), True, type_=bool)
    poly2 = numpoly.polynomial([1e10*Y, 1e-8])
    assert_equal(func_interface.allclose(poly1, poly2), False, type_=bool)


def test_amax(func_interface):
    """Tests for numpoly.amax."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(func_interface.amax(poly), X**4)
    assert_equal(func_interface.amax(poly, axis=0), [X**4, Y**2, X**2, Y**2])
    assert_equal(func_interface.amax(poly, axis=1), [X**2, X**2, Y**2, X**4])
    assert_equal(func_interface.amax(poly.reshape(2, 2, 2, 2), axis=(0, 1)),
                 [[X**4, Y**2], [X**2, Y**2]])


def test_amin(func_interface):
    """Tests for numpoly.amin."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(func_interface.amin(poly), 1)
    assert_equal(func_interface.amin(poly, axis=0), [1, 3, 2, X])
    assert_equal(func_interface.amin(poly, axis=1), [1, 2, 3, Y])
    assert_equal(func_interface.amin(poly.reshape(2, 2, 2, 2), axis=(0, 1)),
                 [[1, 3], [2, X]])


def test_argmax(func_interface):
    """Tests for numpoly.argmax."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(func_interface.argmax(poly), 13)
    assert_equal(func_interface.argmax(poly, axis=0), [3, 3, 1, 0])
    assert_equal(func_interface.argmax(poly, axis=1), [3, 2, 1, 1])


def test_argmin(func_interface):
    """Tests for numpoly.argmin."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(func_interface.argmin(poly), 0)
    assert_equal(func_interface.argmin(poly, axis=0), [0, 0, 2, 1])
    assert_equal(func_interface.argmin(poly, axis=1), [0, 3, 2, 2])


def test_apply_along_axis(func_interface):
    """Tests for numpoly.apply_along_axis."""
    np_array = numpy.arange(9, dtype=int).reshape(3, 3)
    assert_equal(
        func_interface.apply_along_axis(numpy.sum, 0, np_array), [9, 12, 15])
    assert_equal(
        func_interface.apply_along_axis(numpy.sum, 1, np_array), [3, 12, 21])
    poly1 = numpoly.polynomial([[X, X, X], [Y, Y, Y], [1, 2, 3]])
    assert_equal(
        func_interface.apply_along_axis(numpoly.sum, 0, poly1),
        [X+Y+1, X+Y+2, X+Y+3])
    assert_equal(
        func_interface.apply_along_axis(numpoly.sum, 1, poly1), [3*X, 3*Y, 6])


def test_apply_over_axes(func_interface):
    """Tests for numpoly.apply_over_axes."""
    np_array = numpy.arange(9).reshape(3, 3)
    assert_equal(
        func_interface.apply_over_axes(numpy.sum, np_array, 0), [[9, 12, 15]])
    assert_equal(
        func_interface.apply_over_axes(numpy.sum, np_array, 1),
        [[3], [12], [21]])
    poly1 = numpoly.polynomial([[X, X, X], [Y, Y, Y], [1, 2, 3]])
    assert_equal(
        func_interface.apply_over_axes(
            numpoly.sum, poly1, 0), [[X+Y+1, X+Y+2, X+Y+3]])
    assert_equal(
        func_interface.apply_over_axes(
            numpoly.sum, poly1, 1), [[3*X], [3*Y], [6]])


def test_around(interface):
    """Tests for numpoly.around."""
    poly = 123.45*X+Y
    assert_equal(interface.round(poly), 123.*X+Y)
    assert_equal(interface.round(poly, decimals=1), 123.4*X+Y)
    assert_equal(interface.round(poly, decimals=-2), 100.*X)
    out = 5.*X+6.*Y
    interface.round(poly, decimals=1, out=out)
    assert_equal(out, 123.4*X+Y)


def test_array_repr(func_interface):
    """Tests for numpoly.array_repr."""
    assert repr(polynomial([])).startswith("polynomial([], dtype=")
    assert repr(4+6*X**2) == "polynomial(6*q0**2+4)"
    assert func_interface.array_repr(4+6*X**2) == "polynomial(6*q0**2+4)"
    assert (repr(polynomial([1., -5*X, 3-X**2])) ==
            "polynomial([1.0, -5.0*q0, -q0**2+3.0])")
    assert (func_interface.array_repr(polynomial([1., -5*X, 3-X**2])) ==
            "polynomial([1.0, -5.0*q0, -q0**2+3.0])")
    assert repr(polynomial([[[1, 2], [5, Y]]])) == """\
polynomial([[[1, 2],
             [5, q1]]])"""
    assert func_interface.array_repr(polynomial([[[1, 2], [5, Y]]])) == """\
polynomial([[[1, 2],
             [5, q1]]])"""


def test_array_split(func_interface):
    """Tests for numpoly.array_split."""
    test_split(func_interface)
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2 = func_interface.array_split(poly, 2, axis=1)
    assert_equal(part1, [[1, X], [X+Y, Y]])
    assert_equal(part2, [[X**2], [Y]])


def test_array_str(func_interface):
    """Tests for numpoly.array_str."""
    assert str(polynomial([])).startswith("[]")
    assert str(4+6*X**2) == "6*q0**2+4"
    assert func_interface.array_str(4+6*X**2) == "6*q0**2+4"
    assert str(polynomial([1., -5*X, 3-X**2])) == "[1.0 -5.0*q0 -q0**2+3.0]"
    assert func_interface.array_str(
        polynomial([1., -5*X, 3-X**2])) == "[1.0 -5.0*q0 -q0**2+3.0]"
    assert str(polynomial([[[1, 2], [5, Y]]])) == """\
[[[1 2]
  [5 q1]]]"""
    assert func_interface.array_str(polynomial([[[1, 2], [5, Y]]])) == """\
[[[1 2]
  [5 q1]]]"""


def test_atleast_1d(func_interface):
    """Tests for numpoly.atleast_1d."""
    polys = [X, [X], [[X]], [[[X]]]]
    results = func_interface.atleast_1d(*polys)
    assert isinstance(results, list)
    assert_equal(results[0], [X])
    assert_equal(results[1], [X])
    assert_equal(results[2], [[X]])
    assert_equal(results[3], [[[X]]])


def test_atleast_2d(func_interface):
    """Tests for numpoly.atleast_2d."""
    polys = [X, [X], [[X]], [[[X]]]]
    results = func_interface.atleast_2d(*polys)
    assert isinstance(results, list)
    assert_equal(results[0], [[X]])
    assert_equal(results[1], [[X]])
    assert_equal(results[2], [[X]])
    assert_equal(results[3], [[[X]]])


def test_atleast_3d(func_interface):
    """Tests for numpoly.atleast_3d."""
    polys = [X, [X], [[X]], [[[X]]]]
    results = func_interface.atleast_3d(*polys)
    assert isinstance(results, list)
    assert_equal(results[0], [[[X]]])
    assert_equal(results[1], [[[X]]])
    assert_equal(results[2], [[[X]]])
    assert_equal(results[3], [[[X]]])


def test_broadcast_array(func_interface):
    """Tests for numpoly.broadcast_array."""
    polys = [X, [[Y, 1]], [[X], [Y]], [[X, 1], [Y, 2]]]
    results = func_interface.broadcast_arrays(*polys)
    assert isinstance(results, list)
    assert len(results) == len(polys)
    assert all(result.shape == (2, 2) for result in results)
    assert_equal(results[0], [[X, X], [X, X]])
    assert_equal(results[1], [[Y, 1], [Y, 1]])
    assert_equal(results[2], [[X, X], [Y, Y]])
    assert_equal(results[3], [[X, 1], [Y, 2]])


def test_ceil(func_interface):
    """Tests for numpoly.ceil."""
    poly = polynomial([-1.7*X, X-1.5, -0.2, 3.2+1.5*X, 1.7, 2.0])
    assert_equal(func_interface.ceil(poly),
                 [-X, -1.0+X, 0.0, 4.0+2.0*X, 2.0, 2.0])


def test_common_type(func_interface):
    """Tests for numpoly.common_type."""
    assert func_interface.common_type(
        numpy.array(2, dtype=numpy.float32)) == numpy.float32
    assert func_interface.common_type(X) == numpy.float64
    assert func_interface.common_type(
        numpy.arange(3), 1j*X, 45) == numpy.complex128


def test_concatenate(func_interface):
    """Tests for numpoly.concatenate."""
    poly1 = polynomial([[0, Y], [X, 1]])
    assert_equal(func_interface.concatenate([poly1, poly1]),
                 [[0, Y], [X, 1], [0, Y], [X, 1]])
    assert_equal(func_interface.concatenate([poly1, [[X*Y, 1]]], 0),
                 [[0, Y], [X, 1], [X*Y, 1]])
    assert_equal(func_interface.concatenate([poly1, [[X*Y], [1]]], 1),
                 [[0, Y, X*Y], [X, 1, 1]])
    assert_equal(func_interface.concatenate([poly1, poly1], 1),
                 [[0, Y, 0, Y], [X, 1, X, 1]])


def test_copyto(func_interface):
    """Tests for numpoly.copyto."""
    poly = numpoly.polynomial([1, X, Y])
    poly_ref = numpoly.polynomial([1, X, Y])
    with raises(ValueError):
        func_interface.copyto(poly.values, poly_ref, casting="safe")
    with raises(ValueError):
        numpoly.copyto(poly.values, [1, 2, 3], casting="safe")
    with raises(ValueError):
        numpoly.copyto(X, Y, casting="unsafe")
    func_interface.copyto(poly, X)
    assert_equal(poly, [X, X, X])
    func_interface.copyto(poly.values, poly_ref, casting="unsafe")
    assert_equal(poly, poly_ref)
    func_interface.copyto(poly, 4)
    assert_equal(poly, [4, 4, 4])
    func_interface.copyto(poly.values, poly_ref.values, casting="unsafe")
    assert_equal(poly, poly_ref)
    poly = numpoly.polynomial([1, 2, 3])
    func_interface.copyto(poly, [3, 2, 1], casting="unsafe")
    assert_equal(poly, [3, 2, 1])
    func_interface.copyto(
        poly.values, numpoly.polynomial([1, 2, 3]), casting="unsafe")
    assert_equal(poly, [1, 2, 3])
    out = numpy.zeros(3, dtype=float)
    numpoly.copyto(out, poly, casting="unsafe")
    assert_equal(out, [1., 2., 3.])


def test_count_nonzero(func_interface):
    """Tests for numpoly.count_nonzero."""
    poly1 = polynomial([[0, Y], [X, 1]])
    poly2 = polynomial([[0, Y, X, 0, 0], [3, 0, 0, 2, 19]])
    assert_equal(func_interface.count_nonzero(poly1), 3, type_=int)
    assert_equal(func_interface.count_nonzero(poly1, axis=0), [1, 2])
    assert_equal(func_interface.count_nonzero(poly2, axis=0), [1, 1, 1, 1, 1])
    assert_equal(func_interface.count_nonzero(X), 1, type_=int)


def test_cumsum(interface):
    """Tests for numpoly.cumsum."""
    poly1 = polynomial([[0, Y], [X, 1]])
    assert_equal(interface.cumsum(poly1), [0, Y, X+Y, 1+X+Y])
    assert_equal(interface.cumsum(poly1, axis=0), [[0, Y], [X, Y+1]])
    assert_equal(interface.cumsum(poly1, axis=1), [[0, Y], [X, X+1]])


def test_det():
    """Test for numpoly.det."""
    array = [[1, 2], [3, 4]]
    poly = polynomial([[1, Y], [X, 1]])
    assert_equal(numpoly.det([array, poly]), [-2, 1-X*Y])
    assert_equal(numpy.linalg.det(poly), 1-X*Y)
    assert_equal(
        numpoly.det([[1, X, Y], [Y, 1, X], [X, Y, 1]]), X**3+Y**3-3*X*Y+1)


def test_diag(func_interface):
    """Tests for numpoly.diag."""
    poly = polynomial([[1, 2, X], [4, Y, 6], [7, 8, X+Y]])
    assert_equal(func_interface.diag(poly), [1, Y, X+Y])
    assert_equal(func_interface.diag(poly, k=1), [2, 6])
    assert_equal(func_interface.diag(poly, k=-1), [4, 8])

    poly = polynomial([X, Y])
    assert_equal(func_interface.diag(poly), [[X, 0], [0, Y]])
    assert_equal(func_interface.diag(poly, k=1),
                 [[0, X, 0], [0, 0, Y], [0, 0, 0]])
    assert_equal(func_interface.diag(poly, k=-1),
                 [[0, 0, 0], [X, 0, 0], [0, Y, 0]])


def test_diagonal(interface):
    """Tests for numpoly.diagonal."""
    # TODO: return view instead of copy
    poly = polynomial([[1, 2, X], [4, Y, 6], [7, 8, X+Y]])
    assert_equal(interface.diagonal(poly), [1, Y, X+Y])
    assert_equal(interface.diagonal(poly, offset=1), [2, 6])
    assert_equal(interface.diagonal(poly, offset=-1), [4, 8])

    poly = numpoly.monomial(27).reshape(3, 3, 3)
    assert_equal(interface.diagonal(poly, axis1=0, axis2=1),
                 [[1, X**12, X**24],
                  [X, X**13, X**25],
                  [X**2, X**14, X**26]])
    assert_equal(interface.diagonal(poly, axis1=0, axis2=2),
                 [[1, X**10, X**20],
                  [X**3, X**13, X**23],
                  [X**6, X**16, X**26]])
    assert_equal(interface.diagonal(poly, axis1=1, axis2=2),
                 [[1, X**4, X**8],
                  [X**9, X**13, X**17],
                  [X**18, X**22, X**26]])


def test_diff(func_interface):
    """Tests for numpoly.diff."""
    poly = polynomial([[1, 2, X], [4, Y, 6], [7, 8, X+Y]])
    assert_equal(func_interface.diff(poly),
                 [[1, X-2], [Y-4, 6-Y], [1, X+Y-8]])
    assert_equal(func_interface.diff(poly, n=2),
                 [[X-3], [10-2*Y], [X+Y-9]])
    assert_equal(func_interface.diff(poly, axis=0),
                 [[3, Y-2, 6-X], [3, 8-Y, X+Y-6]])
    assert_equal(func_interface.diff(poly, append=X),
                 [[1, X-2, 0], [Y-4, 6-Y, X-6], [1, X+Y-8, -Y]])
    assert_equal(func_interface.diff(poly, prepend=Y),
                 [[1-Y, 1, X-2], [4-Y, Y-4, 6-Y], [7-Y, 1, X+Y-8]])
    assert_equal(func_interface.diff(poly, append=X, prepend=Y),
                 [[1-Y, 1, X-2, 0], [4-Y, Y-4, 6-Y, X-6], [7-Y, 1, X+Y-8, -Y]])


def test_divmod(func_interface):
    """Tests for numpoly.divmod."""
    array = numpy.array([7, 11])
    quotient, remainder = func_interface.divmod(array, 5)
    assert_equal(quotient, [1, 2])
    assert_equal(remainder, [2, 1])
    with raises(numpoly.FeatureNotSupported):
        func_interface.divmod(array, X)
    with raises(numpoly.FeatureNotSupported):
        func_interface.divmod(X, X)


def test_dsplit(func_interface):
    """Tests for numpoly.dsplit."""
    poly = numpoly.polynomial([[[1, X], [X+Y, Y]]])
    part1, part2 = func_interface.dsplit(poly, 2)
    assert_equal(part1, [[[1], [X+Y]]])
    assert_equal(part2, [[[X], [Y]]])


def test_dstack(func_interface):
    """Tests for numpoly.dstack."""
    poly1 = numpoly.polynomial([1, X, 2])
    poly2 = numpoly.polynomial([Y, 3, 4])
    assert_equal(func_interface.dstack([poly1, poly2]),
                 [[[1, Y], [X, 3], [2, 4]]])


def test_ediff1d(func_interface):
    """Tests for numpoly.ediff1d."""
    poly1 = numpoly.polynomial([1, X, 2])
    assert_equal(func_interface.ediff1d(poly1), [X-1, 2-X])
    poly2 = numpoly.polynomial([Y, 3, 4])
    assert_equal(func_interface.ediff1d(poly2), [3-Y, 1])


def test_expand_dims(func_interface):
    """Tests for numpoly.expand_dims."""
    poly1 = numpoly.polynomial([[1, X], [Y, 2], [3, 4]])
    assert func_interface.expand_dims(poly1, axis=0).shape == (1, 3, 2)
    assert func_interface.expand_dims(poly1, axis=1).shape == (3, 1, 2)
    assert func_interface.expand_dims(poly1, axis=2).shape == (3, 2, 1)
    array = numpy.arange(12).reshape(2, 3, 2)
    assert func_interface.expand_dims(array, axis=1).shape == (2, 1, 3, 2)


def test_equal(interface):
    """Tests for numpoly.equal."""
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert_equal(interface.equal(X, X), True)
    assert_equal(interface.equal(X, [X]), [True])
    assert_equal([X] == X, [True])
    assert_equal(interface.equal(X, Y), False)
    assert_equal(([X, 2+Y] == poly), [[False, True], [True, False]])
    assert_equal(interface.equal(numpoly.polynomial([X, 2+Y]), poly),
                 [[False, True], [True, False]])
    assert_equal((X == poly), [[False, False], [True, False]])
    assert_equal(interface.equal(X, poly), [[False, False], [True, False]])
    assert_equal(poly == poly.T, [[True, False], [False, True]])
    assert_equal(interface.equal(poly, poly.T), [[True, False], [False, True]])


def test_floor(func_interface):
    """Tests for numpoly.floor."""
    poly = polynomial([-1.7*X, X-1.5, -0.2, 3.2+1.5*X, 1.7, 2.0])
    assert_equal(func_interface.floor(poly),
                 [-2.0*X, -2.0+X, -1.0, 3.0+X, 1.0, 2.0])


def test_floor_divide(interface):
    """Tests for numpoly.floor_divide."""
    poly = polynomial([[0., 2.*Y], [X, 2.]])
    assert_equal(interface.floor_divide(poly, 2), [[0., Y], [0., 1]])
    assert_equal(interface.floor_divide(poly, [1., 2.]), [[0., Y], [X, 1]])
    assert_equal(interface.floor_divide(
        poly, [[1., 2.], [2., 1.]]), [[0., Y], [0., 2.]])
    with raises(numpoly.FeatureNotSupported):
        interface.floor_divide(poly, X)
    with raises(numpoly.FeatureNotSupported):
        poly.__floordiv__(poly)
    with raises(numpoly.FeatureNotSupported):
        poly.__rfloordiv__(poly)

    out = numpoly.ndpoly(
        exponents=poly.exponents,
        shape=(2, 2),
        names=("q0", "q1"),
        dtype=float,
    )
    numpoly.floor_divide(poly, 2, out=out)
    assert_equal(out, [[0., Y], [0., 1.]])


def test_full(func_interface):
    """Tests for numpoly.full."""
    assert_equal(numpoly.full((3,), X), [X, X, X])
    assert_equal(numpoly.aspolynomial(
        func_interface.full((3,), 1.*X)), [1.*X, X, X])
    assert_equal(numpoly.full((3,), Y, dtype=float), [1.*Y, Y, Y])
    if func_interface is numpy:  # fails in numpy, but only with func dispatch.
        with raises(ValueError, ):
            assert_equal(numpy.full((3,), Y, dtype=float), [1.*Y, Y, Y])
            raise ValueError


def test_full_like(func_interface):
    """Tests for numpoly.full_like."""
    poly = numpoly.polynomial([1, X, 2])
    assert_equal(func_interface.full_like(poly, X), [X, X, X])
    assert_equal(numpoly.full_like([1, X, 2], X), [X, X, X])
    poly = numpoly.polynomial([1., X, 2])
    assert_equal(func_interface.full_like(poly, Y), [1.*Y, Y, Y])


def test_greater(interface):
    """Tests for numpoly.greater."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(interface.greater(poly, X),
                 [[False, False, False, True],
                  [True, True, True, False],
                  [False, True, False, False],
                  [True, True, True, True]])
    assert_equal(interface.greater(poly, Y),
                 [[False, False, False, True],
                  [False, False, True, False],
                  [False, True, False, False],
                  [True, True, False, False]])


def test_greater_equal(interface):
    """Tests for numpoly.greater_equal."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(interface.greater_equal(poly, X),
                 [[False, True, False, True],
                  [True, True, True, False],
                  [False, True, False, True],
                  [True, True, True, True]])
    assert_equal(interface.greater_equal(poly, Y),
                 [[False, False, False, True],
                  [True, False, True, False],
                  [False, True, False, False],
                  [True, True, True, False]])


def test_hsplit(func_interface):
    """Tests for numpoly.hsplit."""
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2, part3 = func_interface.hsplit(poly, 3)
    assert_equal(part1, [[1], [X+Y]])
    assert_equal(part2, [[X], [Y]])
    assert_equal(part3, [[X**2], [Y]])


def test_hstack(func_interface):
    """Tests for numpoly.hstack."""
    poly1 = numpoly.polynomial([1, X, 2])
    poly2 = numpoly.polynomial([Y, 3, 4])
    assert_equal(func_interface.hstack([poly1, poly2]),
                 [1, X, 2, Y, 3, 4])


def test_less(interface):
    """Tests for numpoly.less."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(interface.less(poly, X),
                 [[True, False, True, False],
                  [False, False, False, True],
                  [True, False, True, False],
                  [False, False, False, False]])
    assert_equal(interface.less(poly, Y),
                 [[True, True, True, False],
                  [False, True, False, True],
                  [True, False, True, True],
                  [False, False, False, True]])


def test_less_equal(interface):
    """Tests for numpoly.less_equal."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(interface.less_equal(poly, X),
                 [[True, True, True, False],
                  [False, False, False, True],
                  [True, False, True, True],
                  [False, False, False, False]])
    assert_equal(interface.less_equal(poly, Y),
                 [[True, True, True, False],
                  [True, True, False, True],
                  [True, False, True, True],
                  [False, False, True, True]])


def test_inner(func_interface):
    """Tests for numpoly.inner."""
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert_equal(func_interface.inner(poly1, poly2), Y)


def test_isclose(func_interface):
    """Tests for numpoly.isclose."""
    poly1 = numpoly.polynomial([1e10*X, 1e-7])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-8])
    assert_equal(func_interface.isclose(poly1, poly2), [True, False])
    poly1 = numpoly.polynomial([1e10*X, 1e-8])
    poly2 = numpoly.polynomial([1.00001e10*X, 1e-9])
    assert_equal(func_interface.isclose(poly1, poly2), [True, True])
    poly2 = numpoly.polynomial([1e10*Y, 1e-8])
    assert_equal(func_interface.isclose(poly1, poly2), [False, True])


def test_isfinite(func_interface):
    """Tests for numpoly.isfinite."""
    assert_equal(func_interface.isfinite(X), True)
    assert_equal(func_interface.isfinite(numpy.nan*X), False)
    poly = numpoly.polynomial([numpy.log(-1.), X, numpy.log(0)])
    assert_equal(func_interface.isfinite(poly), [False, True, False])
    out = numpy.ones(3, dtype=bool)
    print(out)
    func_interface.isfinite(poly, out=(out,))
    print(out)
    assert_equal(out, [False, True, False])


def test_logical_and(func_interface):
    """Tests for numpoly.logical_and."""
    poly1 = numpoly.polynomial([0, X])
    poly2 = numpoly.polynomial([1, X])
    poly3 = numpoly.polynomial([0, Y])
    assert_equal(func_interface.logical_and(1, poly1), [False, True])
    assert_equal(1 and poly1, poly1)
    assert_equal(func_interface.logical_and(1, poly2), [True, True])
    assert_equal(1 and poly2, poly2)
    assert_equal(func_interface.logical_and(poly2, poly3), [False, True])


def test_logical_or(func_interface):
    """Tests for numpoly.logical_or."""
    poly1 = numpoly.polynomial([0, X])
    poly2 = numpoly.polynomial([1, X])
    poly3 = numpoly.polynomial([0, Y])
    assert_equal(func_interface.logical_or(1, poly1), [True, True])
    assert_equal(1 or poly1, 1, type_=int)
    assert_equal(func_interface.logical_or(0, poly1), [False, True])
    assert_equal(0 or poly1, poly1)
    assert_equal(func_interface.logical_or(poly2, poly3), [True, True])


def test_matmul(func_interface):
    """Tests for numpoly.matmul."""
    poly1 = numpoly.polynomial([[0, X], [1, Y]])
    poly2 = numpoly.polynomial([X, 2])
    assert_equal(func_interface.matmul(
        poly1, poly2), [[X**2, 2*X], [X+X*Y, 2+2*Y]])
    assert_equal(func_interface.matmul(
        numpy.ones((2, 5, 6, 4)), numpy.ones((2, 5, 4, 3))),
                 4*numpy.ones((2, 5, 6, 3)))
    with raises(ValueError):
        func_interface.matmul(poly1, 4)
    with raises(ValueError):
        func_interface.matmul(3, poly2)


def test_mean(interface):
    """Tests for numpoly.mean."""
    poly = numpoly.polynomial([[1, 2*X], [3*Y+X, 4]])
    assert_equal(interface.mean(poly), 1.25+0.75*Y+0.75*X)
    assert_equal(interface.mean(poly, axis=0), [0.5+1.5*Y+0.5*X, 2.0+X])
    assert_equal(interface.mean(poly, axis=1), [0.5+X, 2.0+1.5*Y+0.5*X])


def test_max(interface):
    """Tests for numpoly.max."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(interface.max(poly), X**4)
    assert_equal(interface.max(poly, axis=0), [X**4, Y**2, X**2, Y**2])
    assert_equal(interface.max(poly, axis=1), [X**2, X**2, Y**2, X**4])
    assert_equal(interface.max(poly.reshape(2, 2, 2, 2), axis=(0, 1)),
                 [[X**4, Y**2], [X**2, Y**2]])


def test_maximum(func_interface):
    """Tests for numpoly.maximum."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(func_interface.maximum(poly, X),
                 [[X, X, X, X**2], [Y, Y-1, Y**2, X],
                  [X, X**2, X, X], [Y**2, X**4, Y, Y-1]])
    assert_equal(func_interface.maximum(poly, Y),
                 [[Y, Y, Y, X**2], [Y, Y, Y**2, Y],
                  [Y, X**2, Y, Y], [Y**2, X**4, Y, Y]])


def test_min(interface):
    """Tests for numpoly.min."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(interface.min(poly), 1)
    assert_equal(interface.min(poly, axis=0), [1, 3, 2, X])
    assert_equal(interface.min(poly, axis=1), [1, 2, 3, Y])
    assert_equal(interface.min(poly.reshape(2, 2, 2, 2), axis=(0, 1)),
                 [[1, 3], [2, X]])


def test_minimum(func_interface):
    """Tests for numpoly.minimum."""
    poly = numpoly.polynomial([[1, X, X-1, X**2],
                               [Y, Y-1, Y**2, 2],
                               [X-1, X**2, 3, X],
                               [Y**2, X**4, Y, Y-1]])
    assert_equal(func_interface.minimum(poly, X),
                 [[1, X, X-1, X], [X, X, X, 2],
                  [X-1, X, 3, X], [X, X, X, X]])
    assert_equal(func_interface.minimum(poly, Y),
                 [[1, X, X-1, Y], [Y, Y-1, Y, 2],
                  [X-1, Y, 3, X], [Y, Y, Y, Y-1]])


def test_moveaxis():
    """Tests for numpoly.moveaxis."""
    # np.moveaxis dispatching doesn't seem to work
    x = numpy.arange(6).reshape(1, 2, 3)
    assert_equal(numpoly.moveaxis(x, 0, -1),
                 [[[0], [1], [2]], [[3], [4], [5]]])
    assert_equal(numpoly.moveaxis(x, [0, 2], [2, 0]),
                 [[[0], [3]], [[1], [4]], [[2], [5]]])


def test_multiply(func_interface):
    """Tests for numpoly.multiply."""
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert_equal(2*poly, [[0, 4+2*Y], [2*X, 4]])
    assert_equal(func_interface.multiply(2, poly), [[0, 4+2*Y], [2*X, 4]])
    assert_equal([X, 1]*poly, [[0, 2+Y], [X*X, 2]])
    assert_equal(func_interface.multiply([X, 1], poly), [[0, 2+Y], [X*X, 2]])
    assert_equal([[X, 1], [Y, 0]]*poly, [[0, 2+Y], [X*Y, 0]])
    assert_equal(
        func_interface.multiply([[X, 1], [Y, 0]], poly), [[0, 2+Y], [X*Y, 0]])


def test_negative(func_interface):
    """Tests for numpoly.negative."""
    poly = polynomial([[X, -Y], [-4, Y]])
    assert_equal(-(X-Y-1), 1-X+Y)
    assert_equal(func_interface.negative(X-Y-1), 1-X+Y)
    assert_equal(func_interface.negative(poly), [[-X, Y], [4, -Y]])


def test_nonzero(interface):
    """Tests for numpoly.nonzero."""
    poly = polynomial([[3*X, 0, 0], [0, 4*Y, 0], [5*X+Y, 6*X, 0]])
    assert_equal(poly[interface.nonzero(poly)], [3*X, 4*Y, 5*X+Y, 6*X])


def test_not_equal(interface):
    """Tests for numpoly.not_equal."""
    poly = polynomial([[0, 2+Y], [X, 2]])
    assert_equal(([X, 2+Y] != poly), [[True, False], [False, True]])
    assert_equal(interface.not_equal(numpoly.polynomial([X, 2+Y]), poly),
                 [[True, False], [False, True]])
    assert_equal((X != poly), [[True, True], [False, True]])
    assert_equal(interface.not_equal(X, poly), [[True, True], [False, True]])
    assert_equal(poly != poly.T, [[False, True], [True, False]])
    assert_equal(
        interface.not_equal(poly, poly.T), [[False, True], [True, False]])


def test_ones_like(func_interface):
    """Tests for numpoly.ones_like."""
    poly = numpoly.polynomial([1, X, 2])
    assert_equal(func_interface.ones_like(poly), [1, 1, 1])
    assert_equal(numpoly.ones_like([1, X, 2]), [1, 1, 1])
    poly = numpoly.polynomial([1., X, 2])
    assert_equal(func_interface.ones_like(poly), [1., 1., 1.])


def test_outer(func_interface):
    """Tests for numpoly.outer."""
    poly1, poly2 = polynomial([[0, Y], [X+1, 1]])
    assert_equal(func_interface.outer(poly1, poly2), [[0, 0], [X*Y+Y, Y]])


def test_positive(func_interface):
    """Tests for numpoly.positive."""
    poly = polynomial([[0, Y], [X, 1]])
    assert_equal(poly, +poly)
    assert_equal(poly, func_interface.positive(poly))
    assert poly is not +poly
    assert poly is not func_interface.positive(poly)


def test_power(func_interface):
    """Tests for numpoly.power."""
    poly = polynomial([[0, Y], [X-1, 2]])
    assert_equal(X**[2], [X**2])
    assert_equal(func_interface.power(X, [2]), [X**2])
    assert_equal(polynomial([X])**[2], [X**2])
    assert_equal(func_interface.power(polynomial([X]), [2]), [X**2])
    assert_equal(polynomial([X, Y])**[2], [X**2, Y**2])
    assert_equal(func_interface.power(polynomial([X, Y]), [2]), [X**2, Y**2])
    assert_equal(polynomial([X])**[1, 2], [X, X**2])
    assert_equal(func_interface.power(polynomial([X]), [1, 2]), [X, X**2])
    assert_equal((X*Y)**[0, 1, 2, 3], [1, X*Y, X**2*Y**2, X**3*Y**3])
    assert_equal(func_interface.power(
        X*Y, [0, 1, 2, 3]), [1, X*Y, X**2*Y**2, X**3*Y**3])
    assert_equal(poly ** 2, [[0, Y**2], [X*X-2*X+1, 4]])
    assert_equal(func_interface.power(poly, 2), [[0, Y**2], [X*X-2*X+1, 4]])
    assert_equal(poly ** [1, 2], [[0, Y**2], [X-1, 4]])
    assert_equal(func_interface.power(poly, [1, 2]), [[0, Y**2], [X-1, 4]])
    assert_equal(poly ** [[1, 2], [2, 1]], [[0, Y**2], [X*X-2*X+1, 2]])
    assert_equal(func_interface.power(
        poly, [[1, 2], [2, 1]]), [[0, Y**2], [X*X-2*X+1, 2]])


def test_prod(interface):
    """Tests for numpoly.prod."""
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    assert_equal(interface.prod(poly), X**3*Y**3+X**4*Y**2)
    assert_equal(interface.prod(poly, axis=0), [Y+X, X*Y, X**2*Y])


def test_remainder(func_interface):
    """Tests for numpoly.remainder."""
    assert_equal(func_interface.remainder([7, 11], 5), [2, 1])
    with raises(numpoly.FeatureNotSupported):
        func_interface.remainder(X, X)
    with raises(numpoly.FeatureNotSupported):
        func_interface.remainder([1, 2], X)


def test_repeat(func_interface):
    """Tests for numpoly.repeat."""
    poly = numpoly.polynomial([[1, X-1], [X**2, X]])
    assert_equal(func_interface.repeat(poly, 2),
                 [[1, -1+X], [1, -1+X], [X**2, X], [X**2, X]])
    assert_equal(func_interface.repeat(poly, 3, axis=1),
                 [[1, 1, 1, -1+X, -1+X, -1+X],
                  [X**2, X**2, X**2, X, X, X]])
    assert_equal(numpoly.repeat(poly, [1, 2], axis=0),
                 [[1, -1+X], [X**2, X], [X**2, X]])


def test_reshape(interface):
    """Tests for numpoly.reshape."""
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    assert_equal(interface.reshape(poly, (3, 2)),
                 [[1, X], [X**2, X+Y], [Y, Y]])
    assert_equal(interface.reshape(poly, 6),
                 [1, X, X**2, X+Y, Y, Y])


def test_result_type(func_interface):
    """Tests for numpoly.result_type."""
    dtypes = ["uint8", "int16", "float32", "complex64"]
    dtypes = [numpy.dtype(dtype) for dtype in dtypes]
    for idx, dtype1 in enumerate(dtypes):
        for dtype2 in dtypes[idx:]:
            assert func_interface.result_type(3, dtype1, dtype2) == dtype2
            assert func_interface.result_type(
                numpoly.variable(dtype=dtype1),
                numpy.arange(3, dtype=dtype2),
            ) == dtype2


def test_rint(func_interface):
    """Tests for numpoly.rint."""
    poly = numpoly.polynomial([-1.7*X, X-1.5])
    assert_equal(func_interface.rint(poly), [-2.*X, X-2.])


def test_roots(func_interface):
    func_interface = numpoly
    poly = [3.2, 2, 1]
    assert_equal(func_interface.roots(poly),
                 [-0.3125+0.46351241j, -0.3125-0.46351241j])
    poly = 3.2*Y**2+2*Y+1
    assert_equal(func_interface.roots(poly),
                 [-0.3125+0.46351241j, -0.3125-0.46351241j])
    with raises(ValueError):
        func_interface.roots(X*Y)
    with raises(ValueError):
        func_interface.roots([[X, 1], [2, X]])


def test_split(func_interface):
    """Tests for numpoly.split."""
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2 = func_interface.split(poly, 2, axis=0)
    assert_equal(part1, [[1, X, X**2]])
    assert_equal(part2, [[X+Y, Y, Y]])
    part1, part2, part3 = func_interface.split(poly, 3, axis=1)
    assert_equal(part1, [[1], [X+Y]])
    assert_equal(part2, [[X], [Y]])
    assert_equal(part3, [[X**2], [Y]])


def test_square(func_interface):
    """Tests for numpoly.square."""
    assert_equal(func_interface.square(X+Y), X**2+2*X*Y+Y**2)
    assert_equal((1+X)**2, 1+2*X+X**2)


def test_stack(func_interface):
    """Tests for numpoly.stack."""
    poly = polynomial([1, X, Y])
    assert_equal(
        func_interface.stack([poly, poly], axis=0), [[1, X, Y], [1, X, Y]])
    assert_equal(
        func_interface.stack([poly, poly], axis=1), [[1, 1], [X, X], [Y, Y]])


def test_subtract(func_interface):
    """Tests for numpoly.subtract."""
    assert_equal(-X+3, 3-X)
    assert_equal(4 - polynomial([1, X, Y]), [3, 4-X, 4-Y])
    assert_equal(
        func_interface.subtract(4, polynomial([1, X, Y])), [3, 4-X, 4-Y])
    assert_equal(polynomial([0, X]) - polynomial([Y, 0]), [-Y, X])
    assert_equal(func_interface.subtract(polynomial([0, X]), [Y, 0]), [-Y, X])
    assert_equal(polynomial([[1, X], [Y, X*Y]]) - [2, X],
                 [[-1, 0], [Y-2, X*Y-X]])
    assert_equal(
        func_interface.subtract(polynomial([[1, X], [Y, X*Y]]), [2, X]),
        [[-1, 0], [Y-2, X*Y-X]])


def test_sum(interface):
    """Tests for numpoly.sum."""
    poly = polynomial([[1, 5*X], [X+3, -Y]])
    assert_equal(interface.sum(poly), -Y+X*6+4)
    assert_equal(interface.sum(poly, axis=0), [X+4, -Y+X*5])
    assert_equal(
        interface.sum(poly, axis=-1, keepdims=True), [[X*5+1], [X-Y+3]])


def test_transpose(func_interface):
    """Tests for numpoly.transpose."""
    poly = numpoly.polynomial([[1, X-1], [X**2, X]])
    assert_equal(func_interface.transpose(poly),
                 [[1, X**2], [X-1, X]])
    assert_equal(
        poly.T, [[1, X**2], [X-1, X]], c_contiguous=False, f_contiguous=True)


def test_true_divide(func_interface):
    """Tests for numpoly.true_divide."""
    poly = polynomial([[0, Y], [X, 1]])
    assert_equal(func_interface.true_divide(
        poly, 2), polynomial([[0, 0.5*Y], [0.5*X, 0.5]]))
    assert_equal(func_interface.true_divide(
        poly, [1, 2]), [[0, 0.5*Y], [X, 0.5]])
    assert_equal(func_interface.true_divide(
        poly, [[1, 2], [2, 1]]), [[0, 0.5*Y], [0.5*X, 1]])
    with raises(numpoly.FeatureNotSupported):
        func_interface.true_divide(poly, X)


def test_vsplit(func_interface):
    """Tests for numpoly.vsplit."""
    poly = numpoly.polynomial([[1, X, X**2], [X+Y, Y, Y]])
    part1, part2 = func_interface.vsplit(poly, 2)
    assert_equal(part1, [[1, X, X**2]])
    assert_equal(part2, [[X+Y, Y, Y]])


def test_vstack(func_interface):
    """Tests for numpoly.vstack."""
    poly1 = numpoly.polynomial([1, X, 2])
    poly2 = numpoly.polynomial([Y, 3, 4])
    assert_equal(func_interface.vstack([poly1, poly2]), [[1, X, 2], [Y, 3, 4]])


def test_where(func_interface):
    """Tests for numpoly.where."""
    poly1 = numpoly.polynomial([0, 4, 0])
    poly2 = numpoly.polynomial([Y, 0, X])
    assert_equal(func_interface.where([0, 1, 0], poly1, poly2), [Y, 4, X])
    assert_equal(func_interface.where([1, 0, 1], poly1, poly2), [0, 0, 0])
    assert_equal(func_interface.where(poly1)[0], [1])
    assert_equal(func_interface.where(poly2)[0], [0, 2])


def test_zeros_like(func_interface):
    """Tests for numpoly.zeros_like."""
    poly = numpoly.polynomial([1, X, 2])
    assert_equal(func_interface.zeros_like(poly), [0, 0, 0])
    assert_equal(numpoly.zeros_like([1, X, 2]), [0, 0, 0])
    poly = numpoly.polynomial([1., X, 2])
    assert_equal(func_interface.zeros_like(poly), [0., 0., 0.])
