"""Return element-wise quotient and remainder simultaneously."""
import numpy
import numpoly


def divmod(dividend, divisor, out=None, where=True, **kwargs):
    """
    Return element-wise quotient and remainder simultaneously.

    ``numpoly.divmod(x, y)`` is equivalent to ``(x / y, x % y)``, but faster
    because it avoids redundant work. It is used to implement the Python
    built-in function ``divmod`` on Numpoly arrays.

    Note that unlike numbers, this returns the polynomial division and
    polynomial remainder. This means that this function is _not_ backwards
    compatible with ``numpy.divmod`` for constants. For example::

        >>> numpy.divmod(11, 2)
        (5, 1)
        >>> numpoly.divmod(11, 2)
        (polynomial(5.5), polynomial(0.0))

    Args:
        dividend (numpoly.ndpoly):
            The array being divided.
        divisor (numpoly.ndpoly):
            Array that that will divide the dividend.
        out (Optional[numpoly.ndpoly]):
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or
            `None`, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of
            outputs.
        where (bool, numpy.ndarray):
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value. Note
            that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        kwargs:
            Keyword args passed to numpy.ufunc.

    Returns:
        (Tuple[numpoly.ndpoly, numpoly.ndpoly]):
            Element-wise quotient and remainder resulting from
            floor division. This is a scalar if both `x1` and `x2`
            are scalars.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> denominator = [x*y**2+2*x**3*y**2, -2+x*y**2]
        >>> numerator = -2+x*y**2
        >>> floor, remainder = numpoly.divmod(denominator, numerator)
        >>> floor
        polynomial([1.0+2.0*x**2, 1.0])
        >>> remainder
        polynomial([2.0+4.0*x**2, 0.0])
        >>> floor*numerator+remainder
        polynomial([x*y**2+2.0*x**3*y**2, -2.0+x*y**2])

    """
    assert where is True, "changing 'where' is not supported."
    dividend, divisor = numpoly.align_polynomials(dividend, divisor)

    if not dividend.shape:
        floor, remainder = divmod(
            dividend.flatten(), divisor.flatten(), out=out, where=where, **kwargs)
        return floor[0], remainder[0]

    quotient = numpoly.zeros(dividend.shape)
    while True:

        candidates = get_division_candidate(dividend, divisor)
        if candidates is None:
            break
        idx1, idx2, include = candidates

        # construct candidate
        candidate = dividend.coefficients[idx1]/numpy.where(
            include, divisor.coefficients[idx2], 1)
        exponent_diff = dividend.exponents[idx1]-divisor.exponents[idx2]
        candidate = candidate*numpoly.prod(divisor.indeterminants**exponent_diff, 0)

        # iterate division algorithm
        quotient = numpoly.add(
            quotient, numpoly.where(include, candidate, 0), **kwargs)
        dividend = numpoly.subtract(
            dividend, numpoly.where(include, divisor*candidate, 0), **kwargs)
        dividend, divisor = numpoly.align_polynomials(dividend, divisor)

    return quotient, dividend


def get_division_candidate(x1, x2):
    """
    Find the next exponent candidate pair in the iterative subtraction process.

    Args:
        x1 (numpoly.ndpoly):
            The array being divided.
        x2 (numpoly.ndpoly):
            Array that that will divide the dividend.

    Returns:
        (Optional[Tuple[int, int, numpy.ndarray]):
            Indices to the exponent candidate in respectively `x1` and `x2`
            that should be used in the divide process, and an boolean array to
            indicate for which coefficients these candidates are valid.

    """
    # Look for exponent candidates among divisors
    for idx2 in reversed(numpy.lexsort(x2.exponents.T)):
        exponent2 = x2.exponents[idx2]

        # Include coefficients where idx2 is non-zero and any potential
        # candidates that is a better fit has coefficient zero. Exponent needs
        # to be the biggest one around.
        include2 = numpy.ones(x2.shape, dtype=bool)
        for idx, exponent in enumerate(x2.exponents):
            if numpy.all(exponent2 <= exponent):
                include2 &= (x2.coefficients[idx] == 0) ^ (idx == idx2)
        if not numpy.any(include2):
            continue

        # Look for exponent candidates among dividend
        for idx1 in reversed(numpy.lexsort(x1.exponents.T)):
            exponent1 = x1.exponents[idx1]

            # Skip situations where numerator has larger exponent than the
            # denominator.
            if numpy.any(exponent1 < exponent2):
                continue

            # Skip if all relevant coefficients of interest are zero.
            include1 = x1.coefficients[idx1] != 0
            include = include1 & include2
            if not numpy.any(include):
                continue

            return idx1, idx2, include

    # No valid candidate pair found; Assume we are done.
    return None