"""Load polynomial or pickled objects from ``.npy``, ``.npz`` or pickled files."""
import numpy
import numpoly


def load(file, mmap_mode=None, allow_pickle=False,
         fix_imports=True, encoding="ASCII"):
    """
    Load polynomial or pickled objects from ``.npy``, ``.npz`` or pickled files.

    Args:
        file (file, str, pathlib.Path):
            The file to read. File-like objects must support the ``seek()``
            and ``read()`` methods. Pickled files require that the file-like
            object support the ``readline()`` method as well.
        mmap_mode (Optional[str]):
            If not None, then memory-map the file, using the given mode (see
            `numpy.memmap` for a detailed description of the modes). A
            memory-mapped array is kept on disk. However, it can be accessed
            and sliced like any ndarray. Memory mapping is especially useful
            for accessing small fragments of large files without reading the
            entire file into memory.
        allow_pickle (bool):
            Allow loading pickled object arrays stored in npy files. Reasons
            for disallowing pickles include security, as loading pickled data
            can execute arbitrary code. If pickles are disallowed, loading
            object arrays will fail.
        fix_imports (bool):
            Only useful when loading Python 2 generated pickled files on
            Python 3, which includes npy/npz files containing object arrays.
            If `fix_imports` is True, pickle will try to map the old Python 2
            names to the new names used in Python 3.
        encoding (Optional[str]):
            What encoding to use when reading Python 2 strings. Only useful
            when loading Python 2 generated pickled files in Python 3, which
            includes npy/npz files containing object arrays. Values other than
            'latin1', 'ASCII', and 'bytes' are not allowed, as they can corrupt
            numerical data.

    Returns:
        (numpoly.ndpoly, Dict[str, numpoly.ndpoly]):
            Data stored in the file. For ``.npz`` files, the returned
            dictionary class must be closed to avoid leaking file
            descriptors.

    Notes:
        For ``.npy`` files (stored with ``numpoly.save``) the polynomial
        indeterminant names are not stored and might not make the save-load
        round trip correctly.
        For ``.npz`` files (stored with ``numpoly.savez``) the return value is
        a Python `dict`, instead of the :class:`numpy.lib.npyio.NpzFile` which
        :func:`numpy.load` returns.

    Examples:
        >>> q0, q1, q2 = numpoly.variable(3)
        >>> poly = numpoly.polynomial([1, q0, q1**2-1])
        >>> numpoly.save("/tmp/123.npy", poly)
        >>> numpoly.load("/tmp/123.npy")
        polynomial([1, q0, q1**2-1])
        >>> mypoly = q2**2-1
        >>> myarray = numpy.array([1, 2, 3])
        >>> numpoly.savez("/tmp/123.npz", mypoly=mypoly, myarray=myarray)
        >>> numpoly.load("/tmp/123.npz")  # doctest: +NORMALIZE_WHITESPACE
        {'myarray': array([1, 2, 3]),
         'mypoly': polynomial(q2**2-1)}

    """
    out = numpy.load(file=file, mmap_mode=mmap_mode,
                     allow_pickle=allow_pickle, fix_imports=fix_imports)
    if isinstance(out, numpy.lib.npyio.NpzFile):
        out = dict(out)
        for key, value in list(out.items()):

            # Classical arrays untouched
            if not value.dtype.names:
                continue

            # Structured array are assumed polynomials
            del out[key]
            for name in value.dtype.names:
                if not name.isdigit():
                    length = len(name)
                    break
            key = key.split("-")
            names = tuple(key[:length])
            key = "-".join(key[length:])
            out[key] = numpoly.polynomial(value, names=names)

    else:
        if out.dtype.names:
            out = numpoly.polynomial(out)
    return out
