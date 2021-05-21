"""Construct symbol variables."""
from typing import Optional, Sequence
import re

import numpy
import numpy.typing

import numpoly
from ..baseclass import ndpoly


def symbols(
        names: Optional[Sequence[str]] = None,
        asarray: bool = False,
        dtype: numpy.typing.DTypeLike = "i8",
        allocation: Optional[int] = None,
) -> ndpoly:
    """
    Construct symbol variables.

    Most directly be providing a list of string names. But a set of shorthands
    also exists:

    * ``,`` and `` `` (space) can be used as a variable delimiter.
    * ``{number}:{number}`` can be used to define a numerical range.
    * ``{letter}:{letter}`` can be used to define a alphabet range.

    Args:
        names:
            Indeterminants are determined by splitting the string on space. If
            iterable of strings, indeterminants defined directly.
        asarray:
            Enforce output as array even in the case where there is only one
            variable.
        dtype:
            The data type of the polynomial coefficients.
        allocation:
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Returns:
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
        coefficients = [numpy.ones(1, dtype=dtype)]
        out = numpoly.ndpoly.from_attributes(
            exponents=[(1,)],
            coefficients=coefficients,
            dtype=dtype,
            allocation=allocation,
        )
        if not asarray:
            out = numpoly.aspolynomial(out[0])
        return out

    if not isinstance(names, str):
        names = tuple(names)

    else:
        names = re.sub(" ", ",", names)
        if "," in names:
            asarray = True
            names = tuple(name for name in names.split(",") if name)

        else:
            match = re.search(r"(\d*):(\d+)", names)
            if match:
                start = int(match.group(1) or 0)
                end = int(match.group(2))
                names = tuple(
                    names.replace(match.group(0), str(idx))
                    for idx in range(start, end)
                )

            else:
                names = (names,)

    assert isinstance(names, tuple)
    exponents = numpy.eye(len(names), dtype=int)
    coefficients = numpy.eye(len(names), dtype=dtype)
    out = numpoly.ndpoly.from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        names=names,
        allocation=allocation,
    )
    if out.size == 1 and not asarray:
        return numpoly.aspolynomial(out[0])
    return out
