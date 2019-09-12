"""Move axes of an array to new positions."""
import numpy
import numpoly

from .common import implements, simple_dispatch


@implements(numpy.moveaxis)
def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions.

    Other axes remain in their original order.

    Args:
        a (numpoly.ndpoly):
            The array whose axes should be reordered.
        source (Union[int, Tuple[int, ...]]):
            Original positions of the axes to move. These must be unique.
        destination (Union[int, Tuple[int, ...]]):
            Destination positions for each of the original axes. These must
            also be unique.

    Returns:
        result (numpoly.ndpoly):
            Array with moved axes. This array is a view of the input array.

    Examples:
        >>> x = numpy.arange(6).reshape(1, 2, 3)
        >>> numpoly.moveaxis(x, 0, -1).shape
        (2, 3, 1)
        >>> numpoly.moveaxis(x, [0, 2], [2, 0]).shape
        (3, 2, 1)

    """
    return simple_dispatch(
        numpy_func=numpy.moveaxis,
        inputs=(a,),
        source=source,
        destination=destination,
    )
