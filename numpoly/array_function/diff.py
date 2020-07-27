"""Calculate the n-th discrete difference along the given axis."""
import numpy
import numpoly

from ..dispatch import implements


@implements(numpy.diff)
def diff(a, n=1, axis=-1, prepend=numpy._NoValue, append=numpy._NoValue):
    """
    Calculate the n-th discrete difference along the given axis.

    The first difference is given by ``out[i] = a[i+1] - a[i]`` along the given
    axis, higher differences are calculated by using `diff` recursively.

    Args:
        a (numpoly.ndpoly):
            Input array
        n (Optional[int]):
            The number of times values are differenced. If zero, the input is
            returned as-is.
        axis (Optional[int]):
            The axis along which the difference is taken, default is the last
            axis.
        prepend, append (Optional[numpy.ndarray]):
            Values to prepend or append to `a` along axis prior to
            performing the difference. Scalar values are expanded to
            arrays with length 1 in the direction of axis and the shape
            of the input array in along all other axes.  Otherwise the
            dimension and shape must match `a` except along axis.

    Returns:
        (numpoly.ndpoly):
            The n-th differences. The shape of the output is the same as `a`
            except along `axis` where the dimension is smaller by `n`. The type
            of the output is the same as the type of the difference between any
            two elements of `a`. This is the same as the type of `a` in most
            cases.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([1, q0, q1, q0**2, q1-1])
        >>> numpoly.diff(poly)
        polynomial([q0-1, q1-q0, q0**2-q1, -q0**2+q1-1])
        >>> numpoly.diff(poly, n=2)
        polynomial([q1-2*q0+1, q0**2-2*q1+q0, -2*q0**2+2*q1-1])
        >>> poly = numpoly.polynomial([[q0, 1], [2, q1]])
        >>> numpoly.diff(poly)
        polynomial([[-q0+1],
                    [q1-2]])
        >>> numpoly.diff(poly, prepend=7, append=q1)
        polynomial([[q0-7, -q0+1, q1-1],
                    [-5, q1-2, 0]])

    """
    if append is not numpy._NoValue:
        if prepend is not numpy._NoValue:
            a, append, prepend = numpoly.align_indeterminants(a, append, prepend)
            a, append, prepend = numpoly.align_exponents(a, append, prepend)
        else:
            a, append = numpoly.align_indeterminants(a, append)
            a, append = numpoly.align_exponents(a, append)
    elif prepend is not numpy._NoValue:
        a, prepend = numpoly.align_indeterminants(a, prepend)
        a, prepend = numpoly.align_exponents(a, prepend)
    else:
        a = numpoly.aspolynomial(a)

    out = None
    for key in a.keys:

        kwargs = {}
        if append is not numpy._NoValue:
            kwargs["append"] = append[key]
        if prepend is not numpy._NoValue:
            kwargs["prepend"] = prepend[key]
        tmp = numpy.diff(a[key], n=n, axis=axis, **kwargs)

        if out is None:
            out = numpoly.ndpoly(
                exponents=a.exponents,
                shape=tmp.shape,
                names=a.indeterminants,
                dtype=tmp.dtype,
            )
        out[key] = tmp

    out = numpoly.clean_attributes(out)
    return out
