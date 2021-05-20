"""Save polynomial array to a binary file in NumPy ``.npy`` format."""
from __future__ import annotations
from os import PathLike

import numpy
import numpoly

from ..baseclass import PolyLike
from ..dispatch import implements


@implements(numpy.save)
def save(
        file: PathLike,
        arr: PolyLike,
        allow_pickle: bool = True,
        fix_imports: bool = True,
) -> None:
    """
    Save polynomial array to a binary file in NumPy ``.npy`` format.

    Args:
        file:
            File or filename to which the data is saved. If file is a
            file-object, then the filename is unchanged. If file is a string
            or Path, a ``.npy`` extension will be appended to the filename if
            it does not already have one.
        arr:
            Array data to be saved.
        allow_pickle:
            Allow saving object arrays using Python pickles. Reasons for
            disallowing pickles include security (loading pickled data can
            execute arbitrary code) and portability (pickled objects may not be
            loadable on different Python installations, for example if the
            stored objects require libraries that are not available, and not
            all pickled data is compatible between Python 2 and Python 3).
        fix_imports:
            Only useful in forcing objects in object arrays on Python 3 to be
            pickled in a Python 2 compatible way. If `fix_imports` is True,
            pickle will try to map the new Python 3 names to the old module
            names used in Python 2, so that the pickle data stream is readable
            with Python 2.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([1, q0, q1**2-1])
        >>> numpoly.save("/tmp/example1.npy", poly)
        >>> numpoly.load("/tmp/example1.npy")
        polynomial([1, q0, q1**2-1])

    """
    if isinstance(file, (str, bytes, PathLike)):
        with open(file, "wb") as src:
            return save(src, arr=arr,
                        allow_pickle=allow_pickle, fix_imports=fix_imports)
    arr = numpoly.aspolynomial(arr)
    numpy.save(file=file, arr=arr.values,
               allow_pickle=allow_pickle, fix_imports=fix_imports)
    numpy.save(file=file, arr=numpy.array(arr.names),
               allow_pickle=allow_pickle, fix_imports=fix_imports)
