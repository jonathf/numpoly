"""Return the string representation of an array."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.array_repr)
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """
    Return the string representation of an array.

    Args:
        arr : ndarray
            Input array.
        max_line_width : int, optional
            Inserts newlines if text is longer than `max_line_width`. Defaults
            to ``numpy.get_printoptions()['linewidth']``.
        precision : int, optional
            Floating point precision. Defaults to
            ``numpy.get_printoptions()['precision']``.
        suppress_small : bool, optional
            Represent numbers "very close" to zero as zero; default is False.
            Very close is defined by precision: if the precision is 8, e.g.,
            numbers smaller (in absolute value) than 5e-9 are represented as
            zero. Defaults to ``numpy.get_printoptions()['suppress']``.

    Returns:
        (str):
            The string representation of an array.

    Examples:
        >>> q0 = numpoly.variable()
        >>> numpoly.array_repr(numpoly.polynomial([1, q0]))
        'polynomial([1, q0])'
        >>> numpoly.array_repr(numpoly.polynomial([]))
        'polynomial([], dtype=int64)'
        >>> numpoly.array_repr(
        ...     numpoly.polynomial([1e-6, 4e-7*q0, 2*q0, 3]),
        ...     precision=4,
        ...     suppress_small=True,
        ... )
        'polynomial([0.0, 0.0, 2.0*q0, 3.0])'

    """
    prefix = "polynomial("
    suffix = ")"
    arr = numpoly.aspolynomial(arr)
    if not arr.size:
        return prefix + "[], dtype=%s" % arr.dtype.name + suffix

    if precision is None:
        precision = numpy.get_printoptions()["precision"]
    if suppress_small is None:
        suppress_small = numpy.get_printoptions()["suppress"]
    arr = to_string(arr, precision=precision, suppress_small=suppress_small)

    return prefix + numpy.array2string(
        numpy.array(arr),
        max_line_width=max_line_width,
        separator=", ",
        formatter={"all": str},
        prefix=prefix,
        suffix=suffix,
    ) + suffix


def to_string(poly, precision=None, suppress_small=None):
    """
    Convert numpoly object to string object, or array of string objects.

    Args:
        poly (numpoly.ndpoly):
            Polynomial object to convert to strings.

    Returns:
        (numpy.ndarray, str):
            If scalar, a string, or if array, numpy.array with string values.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[1, q0**3], [q1-1, -3*q0]])
        >>> string_array = to_string(poly)
        >>> string_array
        [['1', 'q0**3'], ['q1-1', '-3*q0']]
        >>> type(string_array[0][0]) == str
        True

    """
    if poly.shape:
        return [
            to_string(
                poly_, precision=precision, suppress_small=suppress_small)
            for poly_ in poly
        ]

    exponents_ = poly.exponents.copy()
    coefficients = poly.coefficients
    options = numpoly.get_options()
    output = []
    indices = numpoly.glexsort(
        exponents_.T,
        graded=options["display_graded"],
        reverse=options["display_reverse"],
    )
    if options["display_inverse"]:
        indices = indices[::-1]
    for idx in indices:
        coefficient = coefficients[idx]
        exponents = exponents_[idx]

        if not coefficient or (
                suppress_small and
                abs(coefficient) < 10**-precision  # pylint: disable=invalid-unary-operand-type
        ):
            continue

        if coefficient == 1 and any(exponents):
            out = ""
        elif coefficient == -1 and any(exponents):
            out = "-"
        else:
            out = str(coefficient)

        exps_and_names = list(zip(exponents, poly.names))
        for exponent, indeterminant in exps_and_names:
            if exponent:
                if out not in ("", "-"):
                    out += options["display_multiply"]
                out += indeterminant
            if exponent > 1:
                out += options["display_exponent"]+str(exponent)
        if output and float(coefficient) >= 0:
            out = "+"+out
        output.append(out)

    if output:
        return "".join(output)
    return str(numpy.zeros(1, dtype=poly.dtype).item())
