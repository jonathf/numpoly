"""Testing saving to and loading from disk."""
from tempfile import TemporaryFile
import pickle
import pytest

import numpy
import numpoly


X, Y, Z = numpoly.variable(3)
ARRAY = numpy.array([1, 2, 3])
POLY = numpoly.polynomial([1, X, Z**2-1])


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
