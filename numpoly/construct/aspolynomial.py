"""Convert the input to an polynomial array."""
from __future__ import annotations
from typing import Optional, Union, Tuple

import numpy.typing
import numpoly

from ..baseclass import PolyLike, ndpoly


def aspolynomial(
        poly_like: PolyLike,
        names: Union[None, str, Tuple[str, ...], ndpoly] = None,
        dtype: Optional[numpy.typing.DTypeLike] = None,
) -> ndpoly:
    """
    Convert the input to an polynomial array.

    Args:
        poly_like:
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        names:
            Name of the indeterminant variables. If possible to infer from
            ``poly_like``, this argument will be ignored.
        dtype:
            Data type used for the polynomial coefficients.

    Returns:
        Array interpretation of `poly_like`. No copy is performed if the input
        is already an ndpoly with matching indeterminants names and dtype.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.polynomial(q0) is q0
        False
        >>> numpoly.aspolynomial(q0) is q0
        True

    """
    remain = False
    if isinstance(poly_like, numpoly.ndpoly):

        remain = (dtype is None or dtype == poly_like.dtype)
        if names is not None:
            if isinstance(names, numpoly.ndpoly):
                names_ = names.names
            elif isinstance(names, str):
                names_ = (names,)
            else:
                names_ = tuple(names)
            if len(names_) == 1 and len(poly_like.names) > 1:
                names_ = tuple(f"{names_[0]}{idx}"
                               for idx in range(len(poly_like.indeterminants)))
            remain &= names_ == poly_like.names

    if remain:
        return poly_like  # type: ignore
    return numpoly.polynomial(poly_like, names=names, dtype=dtype)
