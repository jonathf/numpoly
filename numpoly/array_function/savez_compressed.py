"""Save several arrays into a single file in uncompressed ``.npz`` format."""
from __future__ import annotations
from os import PathLike

import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.savez_compressed)
def savez_compressed(
    file: PathLike,
    *args: PolyLike,
    **kwargs: PolyLike,
) -> None:
    """
    Save several arrays into a single file in uncompressed ``.npz`` format.

    If arguments are passed in with no keywords, the corresponding variable
    names, in the ``.npz`` file, are 'arr_0', 'arr_1', etc. If keyword
    arguments are given, the corresponding variable names, in the ``.npz``
    file will match the keyword names.

    Args:
        file:
            Either the filename (string) or an open file (file-like object)
            where the data will be saved. If file is a string or a Path, the
            ``.npz`` extension will be appended to the filename if it is not
            already there.
        args:
            Arrays to save to the file. Since it is not possible for Python to
            know the names of the arrays outside `savez_compressed`, the arrays
            will be saved with names "arr_0", "arr_1", and so on. These
            arguments can be any expression.
        kwds:
            Arrays to save to the file. Arrays will be saved in the file with
            the keyword names.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([q0, q1-1])
        >>> array = numpy.array([1, 2])
        >>> numpoly.savez_compressed("/tmp/savez.npz", a=array, p=poly)
        >>> numpoly.load("/tmp/savez.npz")
        {'a': array([1, 2]), 'p': polynomial([q0, q1-1])}
        >>> numpoly.savez_compressed("/tmp/savez.npz", array, poly)
        >>> out = numpoly.load("/tmp/savez.npz")
        >>> out["arr_0"], out["arr_1"]
        (array([1, 2]), polynomial([q0, q1-1]))

    """
    for idx, arg in enumerate(args):
        assert f"arr_{idx}" not in kwargs, "naming conflict"
        kwargs[f"arr_{idx}"] = arg

    polynomials = {
        key: numpoly.aspolynomial(kwargs.pop(key))
        for key, value in list(kwargs.items())
        if isinstance(value, numpoly.ndpoly)
    }
    kwargs.update({"-".join(poly.names)+"-"+key: poly.values
                   for key, poly in polynomials.items()})
    numpy.savez_compressed(file, **kwargs)
