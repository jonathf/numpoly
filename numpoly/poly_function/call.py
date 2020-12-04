"""Evaluate polynomial."""
import logging
import numpy
import numpoly


def call(poly, args=(), kwargs=None):
    """
    Evaluate polynomial by inserting new values in to the indeterminants.

    Equivalent to calling the polynomial or using the ``__call__`` method.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to evaluate.
        args (Sequence[None, int, float, numpy.ndarray, numpoly.ndpoly]):
            Argument to evaluate indeterminants. Ordered positional by
            ``poly.indeterminants``. None values indicate that a variable is
            not to be evaluated, creating a partial evaluation.
        kwargs (Dict[str, Union[int, float, numpy.ndarray, numpoly.ndpoly]]):
            Same as ``args``, but positioned by name.

    Returns:
        (Union[numpy.ndarray, numpoly.ndpoly]):
            Evaluated polynomial. If the resulting array does not contain any
            indeterminants, an array is returned instead of a polynomial.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[q0, q0-1], [q1, q1+q0]])
        >>> numpoly.call(poly)
        polynomial([[q0, q0-1],
                    [q1, q1+q0]])
        >>> poly
        polynomial([[q0, q0-1],
                    [q1, q1+q0]])
        >>> numpoly.call(poly, (1, 0))
        array([[1, 0],
               [0, 1]])
        >>> numpoly.call(poly, (1,), {"q1": [0, 1, 2]})
        array([[[1, 1, 1],
                [0, 0, 0]],
        <BLANKLINE>
               [[0, 1, 2],
                [1, 2, 3]]])
        >>> numpoly.call(poly, (q1,))
        polynomial([[q1, q1-1],
                    [q1, 2*q1]])
        >>> numpoly.call(poly, kwargs={"q1": q0-1, "q0": 2*q1})
        polynomial([[2*q1, 2*q1-1],
                    [q0-1, 2*q1+q0-1]])

    """
    logger = logging.getLogger(__name__)

    # Make sure kwargs contains all args and nothing but indeterminants:
    kwargs = kwargs if kwargs else {}
    args = list(args)
    for arg, indeterminant in zip(args, poly.names):
        if indeterminant in kwargs:
            raise TypeError(
                "multiple values for argument '%s'" % indeterminant)
        kwargs[indeterminant] = arg
    extra_args = [key for key in kwargs if key not in poly.names]
    if extra_args:
        raise TypeError("unexpected keyword argument '%s'" % extra_args[0])

    if not kwargs:
        return poly.copy()

    # Saturate kwargs with values not given:
    for name, indeterminant in zip(poly.names, poly.indeterminants):
        if kwargs.get(name, None) is None:
            kwargs[name] = indeterminant

    # There can only be one shape:
    ones = numpy.ones((), dtype=int)
    for value in kwargs.values():
        ones = ones * numpy.ones(numpoly.polynomial(value).shape, dtype=int)
    shape = poly.shape+ones.shape

    logger.debug("poly shape: %s", poly.shape)
    logger.debug("kwargs common shape: %s", ones.shape)
    logger.debug("output shape: %s", shape)

    # main loop:
    out = 0
    for exponent, coefficient in zip(poly.exponents, poly.coefficients):
        term = ones
        for power, name in zip(exponent, poly.names):
            term = term*kwargs[name]**power
        if isinstance(term, numpoly.ndpoly):
            tmp = numpoly.outer(coefficient, term)
        else:
            tmp = numpy.outer(coefficient, term)
        out = out+tmp.reshape(shape)

    if isinstance(out, numpoly.ndpoly):
        if out.isconstant():
            out = out.tonumpy()
        else:
            out, _ = numpoly.align_indeterminants(out, poly.indeterminants)
    return out
