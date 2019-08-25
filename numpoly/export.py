"""Polynomial string representation."""
import numpy


def to_array(poly, as_type="str"):
    """
    Convert ndpoly object into an array of strings.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to be converted into string array.

    Returns:
        (numpy.ndarray):
            Numpy ndarray containing string representation of each polynomial.

    Example:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([[1, x**3], [y-1, -3-x]])
        >>> print(to_array(poly))
        [['1' 'x**3']
         ['-1+y' '-3-x']]
        >>> print(to_array(poly, as_type="sympy"))
        [[1 x**3]
         [y - 1 -x - 3]]
    """
    if not poly.shape:
        if as_type == "str":
            out = to_string(poly)
        elif as_type == "sympy":
            out = to_sympy(poly)
        return out
    return numpy.array([
        to_array(poly_, as_type=as_type) for poly_ in poly])


def to_sympy(poly):
    """
    Convert numpoly object to sympy object, or array of sympy objects.

    Args:
        poly (numpoly.ndpoly):
            Polynomial object to convert to sympy.

    Returns:
        (numpy.ndarray, sympy.core.expr.Expr):
            If scalar, a sympy expression object, or if array, numpy.array with
            expression object values.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([[1, x**3], [y-1, -3*x]])
        >>> sympy_poly = to_sympy(poly)
        >>> print(sympy_poly)
        [[1 x**3]
         [y - 1 -3*x]]
        >>> type(sympy_poly.item(-1))
        <class 'sympy.core.mul.Mul'>
    """
    if poly.shape:
        return to_array(poly, as_type="sympy")
    from sympy import symbols
    locals_ = dict(zip(poly._indeterminants, symbols(poly._indeterminants)))
    polynomial = eval(to_string(poly), locals_, {})
    return polynomial


def to_string(poly):
    """
    Convert numpoly object to string object, or array of string objects.

    Args:
        poly (numpoly.ndpoly):
            Polynomial object to convert to strings.

    Returns:
        (numpy.ndarray, str):
            If scalar, a string, or if array, numpy.array with string values.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([[1, x**3], [y-1, -3*x]])
        >>> string_array = to_string(poly)
        >>> print(string_array)
        [['1' 'x**3']
         ['-1+y' '-3*x']]
        >>> type(string_array.item(-1)) == str
        True
    """
    if poly.shape:
        return to_array(poly, as_type="str")

    output = []
    for exponents, coefficient in zip(
            poly.exponents.tolist(), poly.coefficients):

        if not coefficient:
            continue

        if coefficient == 1 and any(exponents):
            out = ""
        elif coefficient == -1 and any(exponents):
            out = "-"
        else:
            out = str(coefficient)

        for exponent, indeterminant in zip(exponents, poly._indeterminants):
            if exponent:
                if out not in ("", "-"):
                    out += "*"
                out += indeterminant
            if exponent > 1:
                out += "**"+str(exponent)
        if output and float(coefficient) >= 0:
            out = "+"+out
        output.append(out)

    if output:
        return "".join(output)
    return str(numpy.zeros(1, dtype=poly._dtype).item())
