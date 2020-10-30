"""Load data from a text file."""
import re
PathLike = str
try:
    from os import PathLike
except ImportError:  # pragma: no cover
    pass

import numpy
from numpy.lib.recfunctions import unstructured_to_structured
import numpoly

from .savetxt import HEADER_TEMPLATE

HEADER_REGEX = re.compile(HEADER_TEMPLATE.format(
    version=r"\S+", names=r"(\S+)", keys=r"(\S+)", shape=r"(\S+)"))


def loadtxt(fname, dtype=float, comments="# ", delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding="bytes",
            max_rows=None):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Args:
        fname (str, pathlib.Path, filehandle):
            File, filename, or generator to read.  If the filename extension
            is ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings.
        dtype (numpy.dtype):
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional,
            and each row will be interpreted as an element of the array.  In
            this case, the number of columns used must match the number of
            fields in the data-type.
        comments (Optional[str]):
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter (Optional[str]):
            The string used to separate values. For backwards compatibility,
            byte strings will be decoded as 'latin1'. The default is
            whitespace.
        converters (Optional[Dict]):
            A dictionary mapping column number to a function that will parse
            the column string into the desired value.  E.g., if column 0 is a
            date string: ``converters = {0: datestr2num}``.  Converters can
            also be used to provide a default value for missing data (but see
            also `genfromtxt`): ``converters = {3: lambda s: float(s.strip()
            or 0)}``.
        skiprows (int):
            Skip the first `skiprows` lines, including comments.
        usecols (Union[None, int, Sequence[int]]):
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        unpack (bool):
            If True, the returned array is transposed, so that arguments may
            be unpacked using ``x, y, z = loadtxt(...)``.  When used with a
            structured data-type, arrays are returned for each field.
        ndmin (int):
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed. Legal values:
            0, 1 or 2.
        encoding (Optional[str]):
            Encoding used to decode the inputfile. Does not apply to input
            streams. The special value 'bytes' enables backward compatibility
            workarounds that ensures you receive byte arrays as results if
            possible and passes 'latin1' encoded strings to converters.
            Override this value to receive unicode arrays and pass strings as
            input to converters.  If set to None the system default is used.
            The default value is 'bytes'.
        max_rows (Optional[int]): int, optional
            Read `max_rows` lines of content after `skiprows` lines. The
            default is to read all the lines.

    Returns:
        out : ndarray
            Data read from the text file.

    Examples:
        >>> q0, q1, q2 = numpoly.variable(3)
        >>> poly = numpoly.polynomial([[1, q0], [q0, q2**2-1]])
        >>> numpoly.savetxt("/tmp/poly.txt", poly)
        >>> numpoly.loadtxt("/tmp/poly.txt")
        polynomial([[1.0, q0],
                    [q0, q2**2-1.0]])

    """
    if isinstance(fname, (str, bytes, PathLike)):
        with open(fname) as src:
            header = src.readline()
    else:
        header = fname.readline()
    if isinstance(header, bytes):
        header = header.decode("utf-8")

    array = numpy.loadtxt(fname, dtype=dtype, comments=comments,
                          delimiter=delimiter, converters=converters,
                          skiprows=skiprows, usecols=usecols, unpack=unpack,
                          ndmin=ndmin, max_rows=max_rows, encoding=encoding)

    if header.startswith(comments+"numpoly:"):
        names, keys, shape = re.search(HEADER_REGEX, header).groups()
        names = names.split(",")
        keys = keys.split(",")
        shape = [int(idx) for idx in shape.split(",")]
        dtype = numpy.dtype([(key, array.dtype) for key in keys])
        array = unstructured_to_structured(array, dtype)
        array = numpoly.polynomial(array, names=names).reshape(shape)

    return array
