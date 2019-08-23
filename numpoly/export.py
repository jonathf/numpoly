"""
Polynomial string representation::

    >>> x, y = numpoly.symbols("x y")
    >>> poly = numpoly.polynomial(4+6*y**3)
    >>> print(repr(poly))
    polynomial(4+6*y**3)
    >>> print(poly)
    4+6*y**3
    >>> poly = numpoly.polynomial([1., -5*x, 3-y**2])
    >>> print(repr(poly))
    polynomial([1.0, -5.0*x, 3.0-y**2])
    >>> print(poly)
    [1.0 -5.0*x 3.0-y**2]
    >>> poly = numpoly.polynomial([[[1-4*x, x**2], [y-3, x*y*y]]])
    >>> print(repr(poly))
    polynomial([[[1-4*x, x**2],
                 [-3+y, x*y**2]]])
    >>> print(poly)
    [[[1-4*x x**2]
      [-3+y x*y**2]]]
"""
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
    if poly.shape:
        return to_array(poly, as_type="sympy")
    from sympy import symbols, Poly
    locals_ = dict(zip(poly._indeterminants, symbols(poly._indeterminants)))
    polynomial = eval(to_string(poly), locals_, {})
    return polynomial


def to_string(poly):
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
