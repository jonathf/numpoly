"""Construct symbol variables."""
import re
import string
from six import string_types

import numpy
import numpoly


def symbols(names=None, asarray=False, dtype="i8", allocation=None):
    """
    Construct symbol variables.

    Most directly be providing a list of string names. But a set of shorthands
    also exists:

    * ``,`` and `` `` (space) can be used as a variable delimiter.
    * ``{number}:{number}`` can be used to define a numerical range.
    * ``{letter}:{letter}`` can be used to define a alphabet range.

    Args:
        names (None, str, Tuple[str, ...]):
            Indeterminants are determined by splitting the string on space. If
            iterable of strings, indeterminants defined directly.
        asarray (bool):
            Enforce output as array even in the case where there is only one
            variable.
        dtype (numpy.dtype):
            The data type of the polynomial coefficients.
        allocation (Optional[int]):
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Returns:
        (numpoly.ndpoly):
            Polynomial array with unit components in each dimension.

    Examples:
        >>> numpoly.symbols()
        polynomial(q0)
        >>> numpoly.symbols("q4")
        polynomial(q4)
        >>> numpoly.symbols("q:7")
        polynomial([q0, q1, q2, q3, q4, q5, q6])
        >>> numpoly.symbols("q3:6")
        polynomial([q3, q4, q5])
        >>> numpoly.symbols(["q0", "q3", "q99"])
        polynomial([q0, q3, q99])

    """
    if names is None:
        coefficients = numpy.ones((1, 1) if asarray else 1, dtype=dtype)
        return numpoly.ndpoly.from_attributes(
            exponents=[(1,)],
            coefficients=coefficients,
            dtype=dtype,
            allocation=allocation,
        )

    if not isinstance(names, string_types):
        names = list(names)

    else:
        names = re.sub(" ", ",", names)
        if "," in names:
            asarray = True
            names = [name for name in names.split(",") if name]

        elif re.search(r"\d*:\d+", names):

            match = re.search(r"(\d*):(\d+)", names)
            start = int(match.group(1) or 0)
            end = int(match.group(2))
            names = [
                names.replace(match.group(0), str(idx))
                for idx in range(start, end)
            ]

        else:
            names = [names]

    assert isinstance(names, list)
    exponents = numpy.eye(len(names), dtype=int)
    coefficients = numpy.eye(len(names), dtype=dtype)
    if len(names) == 1 and not asarray:
        coefficients = coefficients[0]
    return numpoly.ndpoly.from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        names=names,
        allocation=allocation,
    )
