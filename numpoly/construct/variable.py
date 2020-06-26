"""Simple constructor to create single variables to create polynomials."""
import numpoly


def variable(dimensions=1, asarray=False, dtype="i8", allocation=None):
    """
    Simple constructor to create single variables to create polynomials.

    Args:
        dimensions (int):
            Number of dimensions in the array.
        asarray (bool):
            Enforce output as array even in the case where there is only one
            variable.
        dtype (numpy.dtype):
            The data type of the polynomial coefficients.
        allocation (Optional[int]):
            The maximum number of polynomial exponents. If omitted, use
            length of exponents for allocation.

    Returns:
        (chaospy.poly.polynomial):
            Polynomial array with unit components in each dimension.

    Examples:
        >>> numpoly.variable()
        polynomial(q0)
        >>> numpoly.variable(3)
        polynomial([q0, q1, q2])

    """
    return numpoly.symbols(
        names="q:%d" % dimensions,
        asarray=asarray,
        dtype=dtype,
        allocation=allocation,
    )
