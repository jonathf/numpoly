"""Load data from a text file."""
from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Union
import re
from os import PathLike

import numpy
import numpy.typing
from numpy.lib.recfunctions import unstructured_to_structured
import numpoly

from .savetxt import HEADER_TEMPLATE
from ..baseclass import ndpoly

HEADER_REGEX = re.compile(HEADER_TEMPLATE.format(
    version=r"\S+", names=r"(\S+)", keys=r"(\S+)", shape=r"(\S+)"))


def loadtxt(
    fname: PathLike,
    dtype: numpy.typing.DTypeLike = float,
    comments: str = "# ",
    delimiter: Optional[str] = None,
    converters: Optional[Dict[int, Callable]] = None,
    skiprows: int = 0,
    usecols: Union[None, int, Sequence[int]] = None,
    unpack: bool = False,
    ndmin: int = 0,
    encoding: str = "bytes",
    max_rows: Optional[int] = None,
) -> ndpoly:
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Args:
        fname:
            File, filename, or generator to read.  If the filename extension
            is ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings.
        dtype:
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional,
            and each row will be interpreted as an element of the array.  In
            this case, the number of columns used must match the number of
            fields in the data-type.
        comments:
            The characters or list of characters used to indicate the start of
            a comment. None implies no comments. For backwards compatibility,
            byte strings will be decoded as 'latin1'. The default is '#'.
        delimiter:
            The string used to separate values. For backwards compatibility,
            byte strings will be decoded as 'latin1'. The default is
            whitespace.
        converters:
            A dictionary mapping column number to a function that will parse
            the column string into the desired value.  E.g., if column 0 is a
            date string: ``converters = {0: datestr2num}``.  Converters can
            also be used to provide a default value for missing data (but see
            also `genfromtxt`): ``converters = {3: lambda s: float(s.strip()
            or 0)}``.
        skiprows:
            Skip the first `skiprows` lines, including comments.
        usecols:
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        unpack:
            If True, the returned array is transposed, so that arguments may
            be unpacked using ``x, y, z = loadtxt(...)``.  When used with a
            structured data-type, arrays are returned for each field.
        ndmin:
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed. Legal values:
            0, 1 or 2.
        encoding:
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
        match = re.search(HEADER_REGEX, header)
        assert match is not None
        groups = match.groups()
        names = tuple(groups[0].split(","))
        keys = groups[1].split(",")
        shape = [int(idx) for idx in groups[2].split(",")]
        dtype = numpy.dtype([(key, array.dtype) for key in keys])
        struct = unstructured_to_structured(array, dtype)
        array = numpoly.polynomial(struct, names=names)
        array = numpoly.reshape(array, shape)

    return array
