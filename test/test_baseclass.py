"""Test ndpoly baseclass functionality."""
import numpy
import numpoly

X, Y = XY = numpoly.symbols("X Y")
EMPTY = numpoly.polynomial([])


def test_scalars():
    assert XY.shape == (2,)
    assert XY.size == 2
    assert X.shape == ()
    assert X.size == 1
    assert EMPTY.shape in [(), (0,)]  # different behavior in py2/3
    assert EMPTY.size == 0

    assert numpy.all(numpy.array(XY.coefficients) == [[1, 0], [0, 1]])
    assert X.coefficients == [1]
    assert EMPTY.coefficients == []

    assert numpy.all(XY.exponents == [[1, 0], [0, 1]])
    assert XY.exponents.shape == (2, 2)
    assert X.exponents == 1
    assert X.exponents.shape == (1, 1)
    assert numpy.all(EMPTY.exponents == 0)
    assert EMPTY.exponents.shape == (1, 1)

    assert numpy.all(XY.indeterminants == XY)
    assert X.indeterminants == X

    assert numpy.all(XY.values == numpy.array(
        [(1, 0), (0, 1)], dtype=[("<;", int), (";<", int)]))
    assert X.values == numpy.array((1,), dtype=[("<", int)])
    assert EMPTY.values.dtype == numpy.dtype([(";", int)])
    assert not EMPTY.values.size

    assert not X.isconstant()
    assert EMPTY.isconstant()

    assert X.todict() == {(1,): 1}
    assert EMPTY.todict() == {}

    assert isinstance(EMPTY.tonumpy(), numpy.ndarray)

    assert X.dtype == int
    assert X.astype(float).dtype == float
    assert EMPTY.dtype == int
    assert EMPTY.astype(float).dtype == float
