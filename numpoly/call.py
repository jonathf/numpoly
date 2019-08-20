"""
Evaluate polynomial::

    >>> from numpoly import polynomial
    >>> poly = polynomial({
    ...     (0, 1): [[1., 2.], [3., 4.]], (1, 0): [[4., 5.], [6., 7.]]})
    >>> print(poly)
    [[q1+4.0q0 2.0q1+5.0q0]
     [3.0q1+6.0q0 4.0q1+7.0q0]]

    # >>> print(poly(1, 0))
    # [6. 8.]
    # >>> print(poly(0, 1))
    # [4. 6.]
    # >>> print(poly(1))
"""
import numpy


def evaluate_polynomial(poly, *args, **kwargs):
    """
    Evaluate polynomial.
    """
    # clean up input:
    args_ = [None]*poly.exponents.shape[-1]
    for idx, arg in enumerate(args):
        key = "q%s" % idx
        if key in kwargs:
            assert arg is None, "mixing positional and kwargs"
            args_[idx] = kwargs.pop(key)
        else:
            args_[idx] = arg

    coefficients = numpy.array(poly.coefficients)
    output = coefficients.flatten()
    exponents = numpy.repeat(poly.exponents.T,
                             coefficients.size // len(poly.exponents), -1).T

    shape = ()
    for idx, arg in enumerate(args_):
        arg = numpy.asarray(arg)
        if len(arg.shape) > len(shape):
            shape = arg.shape
        arg = arg.flatten()

        s, t = numpy.mgrid[:coefficients.size, :len(arg)]
        out = arg[t]**exponents[:, idx][s]
        output = (output.T * out.reshape(len(exponents), *shape).T).T

    output = numpy.sum(output.reshape(numpy.array(poly.coefficients).shape+shape), 0)
    return output
