"""Simple constructor to create single variables to create polynomials."""
import numpoly


def variable(dimensions=1, asarray=False, dtype="i8", allocation=None):
    """
    Construct variables that can be used to construct polynomials.

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
        >>> q0, q1, q2 = numpoly.variable(3)
        >>> q1+1
        polynomial(q1+1)
        >>> numpoly.polynomial([q2**3, q1+q2, 1])
        polynomial([q2**3, q2+q1, 1])

    """
    return numpoly.symbols(
        names="%s:%d" % (numpoly.get_options()["default_varname"], dimensions),
        asarray=asarray,
        dtype=dtype,
        allocation=allocation,
    )
