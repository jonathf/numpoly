"""Convert numpoly.ndpoly to sympy polynomial."""
import numpy


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
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[1, q0**3], [q1-1, -3*q0]])
        >>> sympy_poly = to_sympy(poly)
        >>> sympy_poly
        array([[1, q0**3],
               [q1 - 1, -3*q0]], dtype=object)
        >>> type(sympy_poly.item(-1))
        <class 'sympy.core.mul.Mul'>

    """
    if poly.shape:
        return numpy.array([to_sympy(poly_) for poly_ in poly])
    from sympy import symbols
    locals_ = dict(zip(poly.names, symbols(poly.names)))
    polynomial = eval(str(poly), locals_, {})  # pylint: disable=eval-used
    return polynomial
