"""
Polynomial string representation::

    >>> from numpoly import polynomial
    >>> polynomial({(0,): 4, (1,): 6})
    polynomial(4+6*q0)
    >>> print(polynomial({(0,): 4, (1,): 6}))
    4+6q0
    >>> polynomial({(0,): [1., 0., 3.], (1,): [0., -5., -1.]})
    polynomial([1.0, -5.0*q0, 3.0-q0])
    >>> print(polynomial({(0,): [1., 0., 3.], (1,): [0., -5., -1.]}))
    [1.0 -5.0q0 3.0-q0]
    >>> polynomial({(0,): [[[1., 2.], [3., 4.]]], (1,): [[[4., 5.], [6., 7.]]]})
    polynomial([[[1.0+4.0*q0, 2.0+5.0*q0],
                 [3.0+6.0*q0, 4.0+7.0*q0]]])
    >>> print(polynomial({
    ...     (0,): [[[1., 2.], [3., 4.]]], (1,): [[[4., 5.], [6., 7.]]]}))
    [[[1.0+4.0q0 2.0+5.0q0]
      [3.0+6.0q0 4.0+7.0q0]]]
"""
import numpy

VARNAME = "q"
POWER = "^"
SEP = ""


def construct_string_array(
        poly,
        sep=SEP,
        power=POWER,
        varname=VARNAME,
):
    """
    Convert ndpoly object into an array of strings.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to be converted into string array.
        sep (str):
            String separating coefficients and variables.
        power (str):
            Sring separating variables and exponents.
        varname (str, Iterable[str]):
            Name of the variable(s) to use.

    Returns:
        (numpy.ndarray):
            Numpy ndarray containing string representation of each polynomial.

    Example:
        >>> x, y = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[1, x**3], [y-1, -3-x]])
        >>> print(construct_string_array(poly))
        [['1' 'q0^3']
         ['-1+q1' '-3-q0']]
        >>> print(construct_string_array(
        ...     poly, sep="*", power="**", varname=("x", "y")))
        [['1' 'x**3']
         ['-1+y' '-3-x']]
    """
    if isinstance(varname, str):
        varname = tuple("%s%d" % (varname, idx)
                        for idx in range(poly.exponents.shape[1]))
    array = _construct_string_array(poly, sep, power, varname)
    return numpy.array(array)


def _construct_string_array(
        poly,
        sep=SEP,
        power=POWER,
        varname=VARNAME,
):
    if not poly.shape:
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

            for exponent, varname_ in zip(exponents, varname):
                if exponent:
                    if out not in ("", "-"):
                        out += sep
                    out += varname_
                if exponent > 1:
                    out += power+str(exponent)
            if output and float(coefficient) >= 0:
                out = "+"+out
            output.append(out)

        if output:
            return "".join(output)
        return str(numpy.zeros(1, dtype=poly._dtype).item())

    return [
        construct_string_array(poly_, sep=sep, power=power, varname=varname)
        for poly_ in poly
    ]
