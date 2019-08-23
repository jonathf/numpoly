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


def construct_string_array(poly):
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
        >>> print(construct_string_array(poly))
        [['1' 'x**3']
         ['-1+y' '-3-x']]
    """
    if not poly.shape:
        return as_string(poly)
    return numpy.array([construct_string_array(poly_) for poly_ in poly])


def as_sympa(poly):
    assert not poly.shape


def as_string(poly):
    assert not poly.shape
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

        for exponent, varname_ in zip(exponents, poly._indeterminants):
            if exponent:
                if out not in ("", "-"):
                    out += "*"
                out += varname_
            if exponent > 1:
                out += "**"+str(exponent)
        if output and float(coefficient) >= 0:
            out = "+"+out
        output.append(out)

    if output:
        return "".join(output)
    return str(numpy.zeros(1, dtype=poly._dtype).item())
