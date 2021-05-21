"""Move axes of an array to new positions."""
from __future__ import annotations
from typing import Sequence, Union
import numpy

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements, simple_dispatch


@implements(numpy.moveaxis)
def moveaxis(
    a: PolyLike,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> ndpoly:
    """
    Move axes of an array to new positions.

    Other axes remain in their original order.

    Args:
        a:
            The array whose axes should be reordered.
        source:
            Original positions of the axes to move. These must be unique.
        destination:
            Destination positions for each of the original axes. These must
            also be unique.

    Returns:
        Array with moved axes. This array is a view of the input array.

    Examples:
        >>> poly = numpoly.monomial(6).reshape(1, 2, 3)
        >>> numpoly.moveaxis(poly, 0, -1).shape
        (2, 3, 1)
        >>> numpoly.moveaxis(poly, [0, 2], [2, 0]).shape
        (3, 2, 1)

    """
    return simple_dispatch(
        numpy_func=numpy.moveaxis,
        inputs=(a,),
        source=source,
        destination=destination,
    )
