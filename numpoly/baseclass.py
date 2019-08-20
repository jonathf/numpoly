"""Polynomial base class."""
from __future__ import division
from string import printable  # pylint: disable=no-name-in-module

import numpy

from .call import evaluate_polynomial
from .item import getitem

from . import construct, array_function

FORWARD_DICT = dict(enumerate(numpy.array(list(printable), dtype="S1")))
FORWARD_MAP = numpy.vectorize(FORWARD_DICT.get)
INVERSE_DICT = {value: key for key, value in FORWARD_DICT.items()}
INVERSE_MAP = numpy.vectorize(INVERSE_DICT.get)


class ndpoly(numpy.ndarray):  # pylint: disable=invalid-name
    """
    An array object represents a multidimensional, homogeneous polynomial array
    of fixed-size items. An associated data-type object describes the format of
    each element in the array (its byte-order, how many bytes it occupies in
    memory, whether it is an integer, a floating point number, or something
    else, etc.)

    Arrays should be constructed using `polynomial`, `variable` etc.

    Args:
        keys:
            The exponents in an array like object with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is the
            number of dimensions.
        shape:
            Shape of created array.
        dtype:
            Any object that can be interpreted as a numpy data type.
        kwargs:
            Extra arguments passed to `numpy.ndarray`.

    Examples:
        >>> poly = ndpoly(keys=[(0, 1), (0, 0)], shape=(3,))
        >>> print(poly._keys)
        ['01' '00']
        >>> print(poly.exponents)
        [[0 1]
         [0 0]]
        >>> print(poly.shape)
        (3,)
        >>> poly["00"] = 1, 2, 3
        >>> poly["01"] = 4, 5, 6
        >>> print(numpy.array(list(poly.coefficients)))
        [[4 5 6]
         [1 2 3]]
        >>> print(poly)
        [4q1+1 5q1+2 6q1+3]
        >>> print(poly[0])
        4q1+1
    """

    # =================================================
    # Stuff to get subclassing of ndarray to run smooth
    # =================================================

    __array_priority__ = 16
    _dtype = None
    _keys = None

    def __new__(cls, keys=((0,),), shape=(), dtype=None, **kwargs):
        keys = numpy.array(keys, dtype=int)
        dtype_ = "S%d" % keys.shape[1]
        keys = FORWARD_MAP(keys).flatten()
        keys = numpy.array(keys.view(dtype_), dtype="U")

        dtype = int if dtype is None else dtype
        dtype_ = numpy.dtype([(key, dtype) for key in keys])

        obj = super(ndpoly, cls).__new__(
            cls, shape=shape, dtype=dtype_, **kwargs)
        obj._dtype = dtype  # pylint: disable=protected-access
        obj._keys = keys  # pylint: disable=protected-access
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._keys = getattr(obj, "_keys", None)
        self._dtype = getattr(obj, "_dtype", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = list(inputs)
        for idx, input_ in enumerate(inputs):
            input_ = construct.polynomial(input_)
            if len(input_._keys) != 1 or any(  # pylint: disable=protected-access
                    [key != "0" for key in input_._keys[0]]):  # pylint: disable=protected-access
                inputs[idx] = input_

        if ufunc not in array_function.ARRAY_FUNCTIONS:
            return super(ndpoly, self).__array_ufunc__(
                ufunc, method, *inputs, **kwargs)
        return array_function.ARRAY_FUNCTIONS[ufunc](*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """Dispatch numpy library function calls."""
        if func not in array_function.ARRAY_FUNCTIONS:
            return super(ndpoly, self).__array_function__(
                func, types, args, kwargs)
        if not all(issubclass(type_, ndpoly) for type_ in types):
            return NotImplemented
        return array_function.ARRAY_FUNCTIONS[func](*args, **kwargs)

    # ======================================
    # Properties specific for ndpoly objects
    # ======================================

    @property
    def exponents(self):
        """
        Polynomial exponents.

        Examples:
            >>> x, y = numpoly.variable(2)
            >>> poly = 2*x**4-3*y**2+14
            >>> print(poly)
            14-3q1^2+2q0^4
            >>> print(poly.exponents)
            [[0 0]
             [0 2]
             [4 0]]
        """
        keys = numpy.array(self._keys, dtype="S")
        keys = keys.view("S1").reshape(len(keys), -1)
        return INVERSE_MAP(keys)

    @property
    def coefficients(self):
        """
        Polynomial coefficients.

        Examples:
            >>> x, y = numpoly.variable(2)
            >>> poly = 2*x**4-3*y**2+14
            >>> print(poly)
            14-3q1^2+2q0^4
            >>> print(poly.coefficients)
            [14, -3, 2]
        """
        out = numpy.empty((len(self._keys),) + self.shape, dtype=self._dtype)
        for idx, key in enumerate(self._keys):
            out[idx] = numpy.ndarray.__getitem__(self, key)
        return list(out)

    def todict(self):
        """
        Cast polynomial to dict where keys are exponents and values are
        coefficients.

        Examples:
            >>> x, y = numpoly.variable(2)
            >>> poly = 2*x**4-3*y**2+14
            >>> print(poly)
            14-3q1^2+2q0^4
            >>> print(poly.todict())
            {(0, 0): 14, (0, 2): -3, (4, 0): 2}
        """
        return {tuple(exponent): coefficient
                for exponent, coefficient in zip(
                    self.exponents, self.coefficients)}

    # =============================================
    # Override numpy properties to work with ndpoly
    # =============================================

    def all(self, axis=None, out=None, keepdims=False):
        """Wrapper for numpy.all"""
        return array_function.all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        """Wrapper for numpy.any"""
        return array_function.any(self, axis=axis, out=out, keepdims=keepdims)

    def astype(self, dtype, **kwargs):
        """Wrapper for numpy.astype"""
        coefficients = [coefficient.astype(dtype, **kwargs)
                        for coefficient in self.coefficients]
        return construct.polynomial_from_attributes(
            self.exponents, coefficients)

    # ============================================================
    # Override dunder methods that isn't dealt with by dispatching
    # ============================================================

    def __call__(self, *args):
        """Evaluate polynomial"""
        return evaluate_polynomial(self, *args)

    def __eq__(self, other):
        """Left equality"""
        return array_function.equal(self, other)

    def __getitem__(self, index):
        """Get item."""
        return getitem(self, index)

    def __iter__(self):
        coefficients = numpy.array(list(self.coefficients))
        return iter(construct.polynomial_from_attributes(
            self.exponents, coefficients[:, idx])
                    for idx in range(len(self)))

    def __ne__(self, other):
        """Not equal"""
        return array_function.not_equal(self, other)

    def __pow__(self, other):
        """Left power"""
        return array_function.power(self, other)

    def __repr__(self):
        """Canonical string representation"""
        return array_function.array_repr(self)

    def __str__(self):
        """String representation."""
        return array_function.array_str(self)
