"""Test utility functions."""
from pytest import raises
import numpy
import numpoly


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
    assert numpy.all(numpoly.bindex(start=2, stop=3, dimensions=2, ordering="") ==
                     [[0, 2], [1, 1], [2, 0]])
    assert numpy.all(numpoly.bindex(start=2, stop=4, dimensions=2, cross_truncation=0) ==
                     [[0, 2], [2, 0], [0, 3], [3, 0]])
    assert numpy.all(numpoly.bindex(start=2, stop=4, dimensions=2, cross_truncation=1) ==
                     [[0, 2], [1, 1], [2, 0], [0, 3], [1, 2], [2, 1], [3, 0]])
    assert numpy.all(numpoly.bindex(start=2, stop=4, dimensions=2, cross_truncation=2) ==
                     [[0, 2], [1, 1], [2, 0], [0, 3], [1, 2], [2, 1], [3, 0], [2, 2]])
    assert numpy.all(numpoly.bindex(start=0, stop=2, dimensions=3) ==
                     [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])

    assert numpy.all(numpoly.bindex(start=0, stop=[1, 1, 1]) == [0, 0, 0])


def test_bsort():
    indices = numpy.array([[0, 0, 0, 1, 2, 1],
                           [1, 2, 0, 0, 0, 1]])
    assert numpy.all(numpoly.bsort(indices, ordering="") ==
                     [2, 0, 1, 3, 5, 4])
    assert numpy.all(numpoly.bsort(indices, ordering="G") ==
                     [2, 0, 3, 1, 5, 4])
    assert numpy.all(numpoly.bsort(indices, ordering="R") ==
                     [2, 3, 4, 0, 5, 1])
    assert numpy.all(numpoly.bsort(indices, ordering="I") ==
                     [4, 5, 3, 1, 0, 2])
    assert numpy.all(numpoly.bsort(indices, ordering="GR") ==
                     [2, 3, 0, 4, 5, 1])
    assert numpy.all(numpoly.bsort(indices, ordering="IR") ==
                     [1, 5, 0, 4, 3, 2])
    assert numpy.all(numpoly.bsort(indices, ordering="GI") ==
                     [4, 5, 1, 3, 0, 2])
    assert numpy.all(numpoly.bsort(indices, ordering="GRI") ==
                     [1, 5, 4, 0, 3, 2])

    indices = numpy.array([4, 5, 6, 3, 2, 1])
    assert numpy.all(numpoly.bsort(indices) == [5, 4, 3, 0, 1, 2])
    indices = numpy.array([[4, 5, 6, 3, 2, 1]])
    assert numpy.all(numpoly.bsort(indices) == [5, 4, 3, 0, 1, 2])


def test_cross_truncate():
    indices = numpy.array(numpy.mgrid[:10, :10]).reshape(2, -1).T

    assert not numpy.any(numpoly.cross_truncate(indices, -1, norm=0))
    assert numpy.all(indices[numpoly.cross_truncate(indices, 0, norm=0)].T ==
                     [[0], [0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 1, norm=0)].T ==
                     [[0, 0, 1], [0, 1, 0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 2, norm=0)].T ==
                     [[0, 0, 0, 1, 2], [0, 1, 2, 0, 0]])

    assert not numpy.any(numpoly.cross_truncate(indices, -1, norm=1))
    assert numpy.all(indices[numpoly.cross_truncate(indices, 0, norm=1)].T ==
                     [[0], [0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 1, norm=1)].T ==
                     [[0, 0, 1], [0, 1, 0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 2, norm=1)].T ==
                     [[0, 0, 0, 1, 1, 2], [0, 1, 2, 0, 1, 0]])

    assert not numpy.any(numpoly.cross_truncate(indices, -1, norm=100))
    assert numpy.all(indices[numpoly.cross_truncate(indices, 0, norm=100)].T ==
                     [[0], [0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 1, norm=100)].T ==
                     [[0, 0, 1], [0, 1, 0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 2, norm=100)].T ==
                     [[0, 0, 0, 1, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1]])

    assert not numpy.any(numpoly.cross_truncate(indices, -1, norm=numpy.inf))
    assert numpy.all(indices[numpoly.cross_truncate(indices, 0, norm=numpy.inf)].T ==
                     [[0], [0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 1, norm=numpy.inf)].T ==
                     [[0, 0, 1, 1], [0, 1, 0, 1]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 2, norm=numpy.inf)].T ==
                     [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])

    indices = numpy.array(numpy.mgrid[:10, :10, :10]).reshape(3, -1).T
    assert not numpy.any(numpoly.cross_truncate(indices, -1, 1))
    assert numpy.all(indices[numpoly.cross_truncate(indices, 0, 1)].T == [0, 0, 0])
    assert numpy.all(indices[numpoly.cross_truncate(indices, 1, 1)].T ==
                     [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, [0, 0, 1], 1)].T ==
                     [[0, 0], [0, 0], [0, 1]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, [1, 1, 2], 1)].T ==
                     [[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 2, 0, 0]])
    assert numpy.all(indices[numpoly.cross_truncate(indices, [1, 2, 3], 1)].T ==
                     [[0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 1, 2, 0],
                      [0, 1, 2, 3, 0, 1, 0, 0]])
