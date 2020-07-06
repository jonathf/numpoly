"""Testing saving to and loading from disk."""
from tempfile import TemporaryFile
import pickle
import pytest

import numpy
import numpoly


X, Y, Z = numpoly.variable(3)
ARRAY = numpy.array([1, 2, 3])
POLY = numpoly.polynomial([1, X, Z**2-1])


def test_save(func_interface):
    outfile = TemporaryFile()
    func_interface.save(outfile, X)
    func_interface.save(outfile, ARRAY)
    func_interface.save(outfile, POLY)
    outfile.seek(0)
    assert numpy.all(numpoly.load(outfile) == X)
    assert numpy.all(numpoly.load(outfile) == ARRAY)
    assert numpy.all(numpoly.load(outfile) == POLY)

    with open("/tmp/numpoly_save.npy", "wb") as dst:
        func_interface.save(dst, X)
        func_interface.save(dst, ARRAY)
        func_interface.save(dst, POLY)
    with open("/tmp/numpoly_save.npy", "rb") as src:
        assert numpoly.load(src) == X
        assert numpy.all(numpoly.load(src) == ARRAY)
        assert numpy.all(numpoly.load(src) == POLY)


def test_savez(func_interface):
    outfile = TemporaryFile()
    func_interface.savez(outfile, x=X, array=ARRAY, poly=POLY)
    outfile.seek(0)
    results = numpoly.load(outfile)
    assert results["x"] == X
    assert numpy.all(results["array"] == ARRAY)
    assert numpy.all(results["poly"] == POLY)

    with open("/tmp/numpoly_savez.npy", "wb") as dst:
        func_interface.savez(dst, x=X, array=ARRAY, poly=POLY)
    with open("/tmp/numpoly_savez.npy", "rb") as src:
        results = numpoly.load(src)
    assert results["x"] == X
    assert numpy.all(results["array"] == ARRAY)
    assert numpy.all(results["poly"] == POLY)


def test_savez_compressed(func_interface):
    outfile = TemporaryFile()
    func_interface.savez_compressed(outfile, x=X, array=ARRAY, poly=POLY)
    outfile.seek(0)
    results = numpoly.load(outfile)
    assert results["x"] == X
    assert numpy.all(results["array"] == ARRAY)
    assert numpy.all(results["poly"] == POLY)

    with open("/tmp/numpoly_savezc.npy", "wb") as dst:
        func_interface.savez_compressed(dst, x=X, array=ARRAY, poly=POLY)
    with open("/tmp/numpoly_savezc.npy", "rb") as src:
        results = numpoly.load(src)
    assert results["x"] == X
    assert numpy.all(results["array"] == ARRAY)
    assert numpy.all(results["poly"] == POLY)


def test_savetxt(func_interface):
    outfile = TemporaryFile()
    func_interface.savetxt(outfile, POLY)
    outfile.seek(0)
    assert numpy.all(numpoly.loadtxt(outfile) == POLY)

    with open("/tmp/numpoly_save.npy", "wb") as dst:
        func_interface.savetxt(dst, POLY)
    with open("/tmp/numpoly_save.npy", "rb") as src:
        assert numpy.all(numpoly.loadtxt(src) == POLY)


def test_pickle():
    with open("/tmp/numpoly_pickle.pkl", "wb") as dst:
        pickle.dump(POLY, dst)
    with open("/tmp/numpoly_pickle.pkl", "rb") as src:
        assert numpy.all(pickle.load(src) == POLY)
