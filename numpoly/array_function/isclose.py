"""Return true where two arrays are element-wise equal within a tolerance."""
import numpy
import numpoly

from ..dispatch import implements, simple_dispatch


@implements(numpy.isclose)
def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    """
    Return true where two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers. The
    relative difference (`rtol` * abs(`b`)) and the absolute difference `atol`
    are added together to compare against the absolute difference between `a`
    and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                that are much smaller than one (see Notes).

    Args:
        a, b (numpoly.ndpoly):
            Input arrays to compare.
        rtol (float):
            The relative tolerance parameter (see Notes).
        atol (float):
            The absolute tolerance parameter (see Notes).
        equal_nan (bool):
            Whether to compare NaN's as equal.  If True, NaN's in `a` will be
            considered equal to NaN's in `b` in the output array.

    Returns:
        (numpy.ndarray):
            Returns a boolean array of where `a` and `b` are equal within the
            given tolerance. If both `a` and `b` are scalars, returns a single
            boolean value.

    Notes:
        For finite values, isclose uses the following equation to test whether
        two floating point values are equivalent.

        absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

        Unlike the built-in `math.isclose`, the above equation is not symmetric
        in `a` and `b` -- it assumes `b` is the reference value -- so that
        `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
        the default value of atol is not zero, and is used to determine what
        small values should be considered close to zero. The default value is
        appropriate for expected values of order unity: if the expected values
        are significantly smaller than one, it can result in false positives.
        `atol` should be carefully selected for the use case at hand. A zero
        value for `atol` will result in `False` if either `a` or `b` is zero.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> numpoly.isclose([1e10*q0, 1e-7], [1.00001e10*q0, 1e-8])
        array([ True, False])
        >>> numpoly.isclose([1e10*q0, 1e-8], [1.00001e10*q0, 1e-9])
        array([ True,  True])
        >>> numpoly.isclose([1e10*q0, 1e-8], [1.00001e10*q1, 1e-9])
        array([False,  True])
        >>> numpoly.isclose([q0, numpy.nan],
        ...                 [q0, numpy.nan], equal_nan=True)
        array([ True,  True])

    """
    a, b = numpoly.align_polynomials(a, b)
    out = numpy.ones(a.shape, dtype=bool)
    for key in a.keys:
        out &= numpy.isclose(
            a[key], b[key], atol=atol, rtol=rtol, equal_nan=equal_nan)
    return out
