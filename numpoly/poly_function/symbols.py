"""Construct symbol variables."""
import re
import string

import numpy
import numpoly


def symbols(names=None, asarray=False, dtype="i8"):
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

    Returns:
        (numpoly.ndpoly):
            Polynomial array with unit components in each dimension.

    Examples:
        >>> numpoly.symbols()
        polynomial(q)
        >>> numpoly.symbols("z,")
        polynomial([z])
        >>> numpoly.symbols("z", asarray=True)
        polynomial([z])
        >>> numpoly.symbols(["alpha", "beta"])
        polynomial([alpha, beta])
        >>> numpoly.symbols("x y z")
        polynomial([x, y, z])
        >>> numpoly.symbols("a,b,c")
        polynomial([a, b, c])
        >>> numpoly.symbols("q:7")
        polynomial([q0, q1, q2, q3, q4, q5, q6])
        >>> numpoly.symbols("q3:6")
        polynomial([q3, q4, q5])
        >>> numpoly.symbols("za:f")
        polynomial([za, zb, zc, zd, ze, zf])

    """
    if names is None:
        coefficients = numpy.ones((1, 1) if asarray else 1, dtype=dtype)
        return numpoly.ndpoly.from_attributes(
            exponents=[(1,)], coefficients=coefficients)

    if not isinstance(names, str):
        names = list(names)

    else:
        names = re.sub(" ", ",", names)
        if "," in names:
            asarray = True
            names = [name for name in names.split(",") if name]

        elif re.search(r"(\d*):(\d+)", names):

            match = re.search(r"(\d*):(\d+)", names)
            start = int(match.group(1) or 0)
            end = int(match.group(2))
            names = [
                names.replace(match.group(0), str(idx))
                for idx in range(start, end)
            ]

        elif re.search(r"([a-zA-Z]+):([a-zA-Z]+)", names):

            match = re.search(r"([a-zA-Z]):([a-zA-Z])", names)
            start = string.ascii_letters.index(match.group(1))
            end = string.ascii_letters.index(match.group(2))
            names = [
                names.replace(
                    match.group(0), string.ascii_letters[idy])
                for idy in range(start, end+1)
            ]

    exponents = numpy.eye(len(names), dtype=int)
    coefficients = numpy.eye(len(names), dtype=dtype)
    if len(names) == 1 and not asarray:
        coefficients = coefficients[0]
    return numpoly.ndpoly.from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        names=names,
    )
