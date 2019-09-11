"""Polynomial base class."""
from __future__ import division
from six import string_types

import numpy

from .call import call
from .exponent import exponents_to_keys, keys_to_exponents

from . import construct, array_function, poly_function


class ndpoly(numpy.ndarray):  # pylint: disable=invalid-name
    """
    An array object represents a multidimensional, homogeneous polynomial array
    of fixed-size items. An associated data-type object describes the format of
    each element in the array (its byte-order, how many bytes it occupies in
    memory, whether it is an integer, a floating point number, or something
    else, etc.)

    Arrays should be constructed using `polynomial`, `symbols` etc.

    Args:
        exponents (numpy.ndarray):
            The exponents in an integer array with shape ``(N, D)``, where
            ``N`` is the number of terms in the polynomial sum and ``D`` is the
            number of dimensions.
        shape (Tuple[int, ...]):
            Shape of created array.
        indeterminants (Union[str, Tuple[str], numpoly.ndpoly]):
            The name of the indeterminant variables in te polynomial. If
            polynomial, inherent from it. Else, pass argument to
            `numpoly.symbols` to create the indeterminants. If only one name is
            provided, but more than one is required, indeterminant will be
            extended with an integer index.
        dtype:
            Any object that can be interpreted as a numpy data type.
        kwargs:
            Extra arguments passed to `numpy.ndarray` constructor.

    Examples:
        >>> poly = ndpoly(
        ...     exponents=[(0, 1), (0, 0)], shape=(3,), indeterminants="x y")
        >>> poly["00"] = 1, 2, 3
        >>> poly["01"] = 4, 5, 6
        >>> print(numpy.array(list(poly.coefficients)))
        [[4 5 6]
         [1 2 3]]
        >>> print(poly)
        [4*y+1 5*y+2 6*y+3]
        >>> print(poly[0])
        4*y+1
    """

    # =================================================
    # Stuff to get subclassing of ndarray to run smooth
    # =================================================

    __array_priority__ = 16
    _dtype = None
    keys = None
    names = None

    def __new__(
            cls,
            exponents=((0,),),
            shape=(),
            indeterminants="z",
            dtype=None,
            **kwargs
    ):
        exponents = numpy.array(exponents, dtype=int)

        if isinstance(indeterminants, str):
            indeterminants = poly_function.symbols(indeterminants)
        if isinstance(indeterminants, ndpoly):
            indeterminants = indeterminants.names
        if len(indeterminants) == 1 and exponents.shape[1] > 1:
            indeterminants = tuple(
                "%s%d" % (str(indeterminants[0]), idx)
                for idx in range(exponents.shape[1])
            )

        keys = exponents_to_keys(exponents)

        dtype = int if dtype is None else dtype
        dtype_ = numpy.dtype([(key, dtype) for key in keys])

        obj = super(ndpoly, cls).__new__(
            cls, shape=shape, dtype=dtype_, **kwargs)
        obj._dtype = numpy.dtype(dtype)  # pylint: disable=protected-access
        obj.keys = keys  # pylint: disable=protected-access
        obj.names = tuple(indeterminants)  # pylint: disable=protected-access
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.keys = getattr(obj, "keys", None)
        self.names = getattr(obj, "names", None)
        self._dtype = getattr(obj, "_dtype", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # inputs = list(inputs)
        # for idx, input_ in enumerate(inputs):
        #     input_ = construct.aspolynomial(input_)
        #     if len(input_.keys) != 1 or any(  # pylint: disable=protected-access
        #             [key != "0" for key in input_.keys[0]]):  # pylint: disable=protected-access
        #         inputs[idx] = input_

        if ufunc not in array_function.ARRAY_FUNCTIONS:
            return super(ndpoly, self).__array_ufunc__(
                ufunc, method, *inputs, **kwargs)
        return array_function.ARRAY_FUNCTIONS[ufunc](*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in array_function.ARRAY_FUNCTIONS:
            return super(ndpoly, self).__array_function__(
                func, types, args, kwargs)
        # if not all(issubclass(type_, ndpoly) for type_ in types):
        #     return NotImplemented
        return array_function.ARRAY_FUNCTIONS[func](*args, **kwargs)

    # ======================================
    # Properties specific for ndpoly objects
    # ======================================

    @property
    def coefficients(self):
        """
        Polynomial coefficients.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([2*x**4, -3*y**2+14])
            >>> print(poly)
            [2*x**4 14-3*y**2]
            >>> print(numpy.array(poly.coefficients))
            [[ 0 14]
             [ 0 -3]
             [ 2  0]]
        """
        out = numpy.empty((len(self.keys),) + self.shape, dtype=self._dtype)
        for idx, key in enumerate(self.keys):
            out[idx] = numpy.ndarray.__getitem__(self, key)
        return list(out)

    @property
    def exponents(self):
        """
        Polynomial exponents.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([2*x**4, -3*y**2+14])
            >>> print(poly)
            [2*x**4 14-3*y**2]
            >>> print(poly.exponents)
            [[0 0]
             [0 2]
             [4 0]]
        """
        return keys_to_exponents(self.keys)

    @staticmethod
    def from_attributes(
            exponents,
            coefficients,
            indeterminants="q",
            dtype=None,
            clean=True,
    ):
        """
        Construct polynomial from polynomial attributes.

        Args:
            exponents (numpy.ndarray):
                The exponents in an integer array with shape ``(N, D)``, where
                ``N`` is the number of terms in the polynomial sum and ``D`` is
                the number of dimensions.
            coefficients (Iterable[numpy.ndarray]):
                The polynomial coefficients. Must correspond to `exponents` by
                having the same length ``N``.
            indeterminants (Union[str, Tuple[str, ...], numpoly.ndpoly]):
                The indeterminants variables, either as string names or as
                simple polynomials. Must correspond to the exponents by having
                length ``D``.
            dtype (Optional[numpy.dtype]):
                The data type of the polynomial. If omitted, extract from
                `coefficients`.
            clean (bool):
                Clean up attributes, removing redundant indeterminants and
                exponents. Used to ensure alignment isn't broken.

        Examples:
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0]], coefficients=[4])
            polynomial(4)
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[1]], coefficients=[[1, 2, 3]])
            polynomial([q, 2*q, 3*q])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0], [1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q, 1+q])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0, 1], [1, 1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q0*q1, q1+q0*q1])
        """
        return construct.polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
            dtype=dtype,
            clean=clean,
        )

    @property
    def indeterminants(self):
        """
        Polynomial indeterminants.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([2*x**4, -3*y**2+14])
            >>> print(poly)
            [2*x**4 14-3*y**2]
            >>> print(poly.indeterminants)
            [x y]
        """
        return construct.polynomial_from_attributes(
            exponents=numpy.eye(len(self.names), dtype=int),
            coefficients=numpy.eye(len(self.names), dtype=int),
            indeterminants=self.names,
        )

    def isconstant(self):
        """Check if a polynomial is constant or not."""
        return poly_function.isconstant(self)

    def todict(self):
        """
        Cast polynomial to dict where keys are exponents and values are
        coefficients.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = 2*x**4-3*y**2+14
            >>> print(poly)
            14-3*y**2+2*x**4
            >>> print(poly.todict())
            {(0, 0): 14, (0, 2): -3, (4, 0): 2}
        """
        return {tuple(exponent): coefficient
                for exponent, coefficient in zip(
                    self.exponents, self.coefficients)}

    def toarray(self):
        """Cast polynomial to numpy.ndarray, if possible."""
        return poly_function.toarray(self)

    # =============================================
    # Override numpy properties to work with ndpoly
    # =============================================

    @property
    def dtype(self):
        """Show coefficient dtype instead of the structured array"""
        return self._dtype

    def all(self, axis=None, out=None, keepdims=False):
        return array_function.all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        return array_function.any(self, axis=axis, out=out, keepdims=keepdims)

    def astype(self, dtype, **kwargs):
        coefficients = [coefficient.astype(dtype, **kwargs)
                        for coefficient in self.coefficients]
        return construct.polynomial_from_attributes(
            self.exponents, coefficients, self.names)

    def flatten(self):
        coefficients = [coefficient.flatten()
                        for coefficient in self.coefficients]
        return construct.polynomial_from_attributes(
            self.exponents, coefficients, self.names)

    # ============================================================
    # Override dunder methods that isn't dealt with by dispatching
    # ============================================================

    def __call__(self, *args, **kwargs):
        """Evaluate polynomial"""
        return call(self, *args, **kwargs)

    def __eq__(self, other):
        """Left equality"""
        return array_function.equal(self, other)

    def __getitem__(self, index):
        """
        Get array item or slice.

        Args:
            index (Union[int, str, Tuple[int, ...], numpy.ndarray]):
                The index to extract. If string type, extract using th
                underlying structured array data type. Else, extract as
                numpy.ndarray.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([[1-4*x, x**2], [y-3, x*y*y]])
            >>> print(poly)
            [[1-4*x x**2]
             [-3+y x*y**2]]
            >>> print(poly[0])
            [1-4*x x**2]
            >>> print(poly[:, 1])
            [x**2 x*y**2]
            >>> print(poly["10"])
            [[-4  0]
             [ 0  0]]
        """
        if isinstance(index, string_types):
            return numpy.asarray(super(ndpoly, self).__getitem__(index))
        return construct.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=[coeff[index] for coeff in self.coefficients],
            indeterminants=self.names,
        )

    def __iter__(self):
        coefficients = numpy.array(list(self.coefficients))
        return iter(construct.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=coefficients[:, idx],
            indeterminants=self.names,
        ) for idx in range(len(self)))

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