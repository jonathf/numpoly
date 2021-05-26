"""Apply a function to 1-D slices along the given axis."""
from __future__ import annotations
from functools import wraps
from typing import Any, Callable, List

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
from ..dispatch import implements


@implements(numpy.apply_along_axis)
def apply_along_axis(
    func1d: Callable[[PolyLike], PolyLike],
    axis: int,
    arr: PolyLike,
    *args: Any,
    **kwargs: Any,
) -> ndpoly:
    """
    Apply a function to 1-D slices along the given axis.

    Execute `func1d(a, *args)` where `func1d` operates on 1-D arrays and `a` is
    a 1-D slice of `arr` along `axis`.

    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                f = func1d(arr[ii+s_[:,]+kk])
                Nj = f.shape
                for jj in ndindex(Nj):
                    out[ii+jj+kk] = f[jj]

    Equivalently, eliminating the inner loop, this can be expressed as::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                out[ii+s_[...,]+kk] = func1d(arr[ii+s_[:,]+kk])

    Args:
        func1d:
            This function should accept 1-D arrays. It is applied to 1-D slices
            of `arr` along the specified axis.
        axis:
            Axis along which `arr` is sliced.
        arr:
            Input array.
        args:
            Additional arguments to `func1d`.
        kwargs:
            Additional named arguments to `func1d`.

    Returns:
        The output array. The shape of `out` is identical to the shape of
        `arr`, except along the `axis` dimension. This axis is removed, and
        replaced with new dimensions equal to the shape of the return value of
        `func1d`. So if `func1d` returns a scalar `out` will have one fewer
        dimensions than `arr`.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> b = numpoly.polynomial([[1, 2, 3*q0],
        ...                         [3, 6*q1, 6],
        ...                         [2, 7, 9]])
        >>> numpoly.apply_along_axis(numpoly.mean, 0, b)
        polynomial([2.0, 2.0*q1+3.0, q0+5.0])
        >>> numpoly.apply_along_axis(numpoly.mean, 1, b)
        polynomial([q0+1.0, 2.0*q1+3.0, 6.0])

    """
    collection: List[ndpoly] = list()

    @wraps(func1d)
    def wrapper_func(array):
        """Wrap func1d function."""
        # Align indeterminants in case slicing changed them
        array = numpoly.polynomial(
            array, names=arr.indeterminants, allocation=arr.allocation)
        array, _ = numpoly.align.align_indeterminants(
            array, arr.indeterminants)

        # Evaluate function
        out = func1d(array, *args, **kwargs)

        # Restore indeterminants in case func1d changed them.
        out, _ = numpoly.align.align_indeterminants(out, arr.indeterminants)

        # Return dummy index integer value that will be replaced with
        # polynomials afterwards.
        ret_val = len(collection)*numpy.ones(out.shape, dtype=int)
        collection.append(out)
        return ret_val

    # Initiate wrapper
    arr = numpoly.aspolynomial(arr)
    out = numpy.apply_along_axis(wrapper_func, axis=axis, arr=arr.values)

    # align exponents
    polynomials = numpoly.align.align_exponents(*collection)
    dtype = numpoly.result_type(*polynomials)

    # Store results into new array
    ret_val = numpoly.ndpoly(
        exponents=polynomials[0].exponents,
        shape=out.shape,
        names=polynomials[0].indeterminants,
        dtype=dtype,
    ).values
    for idx, polynomial in enumerate(polynomials):
        ret_val[out == idx] = polynomial.values

    return numpoly.polynomial(
        ret_val,
        dtype=dtype,
        names=polynomials[0].indeterminants,
        allocation=polynomials[0].allocation,
    )
