"""Apply a function repeatedly over multiple axes."""
from functools import wraps
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.apply_over_axes)
def apply_over_axes(func, a, axes):
    """
    Apply a function repeatedly over multiple axes.

    `func` is called as `res = func(a, axis)`, where `axis` is the first
    element of `axes`.  The result `res` of the function call must have either
    the same dimensions as `a` or one less dimension.  If `res` has one less
    dimension than `a`, a dimension is inserted before `axis`.  The call to
    `func` is then repeated for each axis in `axes`, with `res` as the first
    argument.

    Args:
        func (Callable):
            This function must take two arguments, `func(a, axis)`.
        a (numpoly.ndpoly):
            Input array.
        axes (numpy.ndarray, int):
            Axes over which `func` is applied; the elements must be integers.

    Returns:
        (numpoly.ndpoly):
            The output array.  The number of dimensions is the same as `a`, but
            the shape can be different.  This depends on whether `func` changes
            the shape of its output with respect to its input.

    Examples:
        >>> polynomial = numpoly.symbols("x y z")*numpy.arange(24).reshape(2, 4, 3)
        >>> numpoly.apply_over_axes(numpoly.sum, polynomial, 1)
        polynomial([[[18*x, 22*y, 26*z]],
        <BLANKLINE>
                    [[66*x, 70*y, 74*z]]])
        >>> numpoly.apply_over_axes(numpoly.sum, polynomial, [0, 2])
        polynomial([[[16*z+14*y+12*x],
                     [22*z+20*y+18*x],
                     [28*z+26*y+24*x],
                     [34*z+32*y+30*x]]])

    """
    @wraps(func)
    def wrapper_func(array, axis):
        """Wrap func function."""
        # Align indeterminants in case slicing changed them
        array = numpoly.polynomial(array, names=a.indeterminants)
        array, _ = numpoly.align.align_indeterminants(array, a.indeterminants)
        # Evaluate function
        out = func(array, axis=axis)
        # Restore indeterminants in case func changed them.
        out, _ = numpoly.align.align_indeterminants(out, a.indeterminants)
        return out

    # Initiate wrapper
    a = numpoly.aspolynomial(a)
    out = numpy.apply_over_axes(wrapper_func, a=a.values, axes=axes)
    return out
