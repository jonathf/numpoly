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
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([[1, x**3], [y-1, -3*x]])
        >>> sympy_poly = to_sympy(poly)
        >>> sympy_poly
        array([[1, x**3],
               [y - 1, -3*x]], dtype=object)
        >>> type(sympy_poly.item(-1))
        <class 'sympy.core.mul.Mul'>

    """
    if poly.shape:
        return numpy.array([to_sympy(poly_) for poly_ in poly])
    from sympy import symbols
    locals_ = dict(zip(poly.names, symbols(poly.names)))
    polynomial = eval(str(poly), locals_, {})  # pylint: disable=eval-used
    return polynomial
