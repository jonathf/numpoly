"""Save polynomial array to a binary file in NumPy ``.npy`` format."""
import logging
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.save)
def save(file, arr, allow_pickle=False, fix_imports=True):
    """
    Save polynomial array to a binary file in NumPy ``.npy`` format.

    Args:
        file (file, str, pathlib.Path):
            File or filename to which the data is saved. If file is a
            file-object, then the filename is unchanged. If file is a string
            or Path, a ``.npy`` extension will be appended to the filename if
            it does not already have one.
        arr (numpoly.ndpoly, Iterable[numpoly.ndpoly]):
            Array data to be saved.
        allow_pickle (bool):
            Allow saving object arrays using Python pickles. Reasons for
            disallowing pickles include security (loading pickled data can
            execute arbitrary code) and portability (pickled objects may not be
            loadable on different Python installations, for example if the
            stored objects require libraries that are not available, and not
            all pickled data is compatible between Python 2 and Python 3).
        fix_imports (bool):
            Only useful in forcing objects in object arrays on Python 3 to be
            pickled in a Python 2 compatible way. If `fix_imports` is True,
            pickle will try to map the new Python 3 names to the old module
            names used in Python 2, so that the pickle data stream is readable
            with Python 2.

    Note:
        The polynomial indeterminant names are not stored with the stored
        array. They need to be provided on load, or default names will be
        used instead. For storing polynomials with names intact, use
        ``numpoly.savez`` instead.

        Unlike numpy's save interface, `allow_pickle` is set to False. This
        is because the discrepancy of this parameter in save and load is
        causing an error. User is free to turn it on again manually, but then
        would have to do the same when loading.

    Examples:
        >>> # Normal usage
        >>> q0, q1, q2 = numpoly.variable(3)
        >>> poly = numpoly.polynomial([1, q0, q1**2-1])
        >>> numpoly.save("/tmp/example1.npy", poly)
        >>> numpoly.load("/tmp/example1.npy")
        polynomial([1, q0, q1**2-1])
        >>> # Backwards compatibility
        >>> array = numpy.array([1, 2, 3])
        >>> numpoly.save("/tmp/example2.npy", array)
        >>> numpoly.load("/tmp/example2.npy")
        polynomial([1, 2, 3])
        >>> # Round-trip not preserved
        >>> poly = q2**2-1
        >>> numpoly.save("/tmp/example3.npy", poly)
        >>> numpoly.load("/tmp/example3.npy")
        polynomial(q0**2-1)

    """
    logger = logging.getLogger(__name__)
    arr = numpoly.aspolynomial(arr)
    default_names = numpoly.variable(len(arr.names)).names
    if arr.names != default_names and not allow_pickle:
        logger.warning(
            "polynomial indeterminant names not aligned with the defaults,"
            "and will not be restored correctly with ``numpoly.load``."
        )
        logger.warning(
            "Use ``pickle`` or ``numpoly.savez`` to preserve round-trip."
        )
    numpy.save(file=file, arr=arr.values, allow_pickle=allow_pickle, fix_imports=fix_imports)
