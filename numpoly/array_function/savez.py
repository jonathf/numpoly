"""Save several arrays into a single file in uncompressed ``.npz`` format."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.savez)
def savez(file, *args, **kwargs):
    """
    Save several arrays into a single file in uncompressed ``.npz`` format.

    If arguments are passed in with no keywords, the corresponding variable
    names, in the ``.npz`` file, are 'arr_0', 'arr_1', etc. If keyword
    arguments are given, the corresponding variable names, in the ``.npz``
    file will match the keyword names.

    Args:
        file (str, file):
            Either the filename (string) or an open file (file-like object)
            where the data will be saved. If file is a string or a Path, the
            ``.npz`` extension will be appended to the filename if it is not
            already there.
        args (numpoly.ndpoly, numpy.ndarray):
            Arrays to save to the file. Since it is not possible for Python to
            know the names of the arrays outside `savez`, the arrays will be
            saved with names "arr_0", "arr_1", and so on. These arguments can
            be any expression.
        kwds (numpoly.ndpoly, numpy.ndarray):
            Arrays to save to the file. Arrays will be saved in the file with
            the keyword names.

    Notes:
        If opened with :func:`numpy.load` instead of
        :func:`~numpoly.array_function.load.load`, the polynomials keys will
        contain extra indeterminant names, and values are structured arrays.
        Full round trip can be completed by casting with
        :func:`~numpoly.construct.polynomial.polynomial` and using the extra
        key information.

    Examples:
        >>> q0, q1, q2 = numpoly.variable(3)
        >>> poly = numpoly.polynomial([1, q0, q2**2-1])
        >>> numpoly.savez("/tmp/savez.npz", poly)
        >>> numpoly.load("/tmp/savez.npz")
        {'arr_0': polynomial([1, q0, q2**2-1])}
        >>> raw = dict(numpy.load("/tmp/savez.npz"))
        >>> raw
        {'q0-q2-arr_0': array([( 1, 0, 0), ( 0, 0, 1), (-1, 1, 0)],
              dtype=[(';;', '<i8'), (';=', '<i8'), ('<;', '<i8')])}
        >>> numpoly.polynomial(raw["q0-q2-arr_0"], names=("q0", "q2"))
        polynomial([1, q0, q2**2-1])

    """
    for idx, arg in enumerate(args):
        assert "arr_%d" % idx not in kwargs, "naming conflict"
        kwargs["arr_%d" % idx] = arg

    polynomials = {
        key: kwargs.pop(key) for key, value in list(kwargs.items())
        if isinstance(value, numpoly.ndpoly)
    }
    polynomials = {"-".join(poly.names)+"-"+key: poly.values
                   for key, poly in polynomials.items()}
    numpy.savez(file, **kwargs, **polynomials)
