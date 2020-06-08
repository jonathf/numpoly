"""Polynomial base class."""
from __future__ import division
import re
from six import string_types

import numpy

from . import construct, dispatch, array_function, poly_function, option


REDUCE_MAPPINGS = {
    numpy.add: numpy.sum,
    numpy.multiply: numpy.prod,
    numpy.logical_and: numpy.all,
    numpy.logical_or: numpy.any,
    numpy.maximum: numpy.amax,
    numpy.minimum: numpy.amin,
}
ACCUMULATE_MAPPINGS = {
    numpy.add: numpy.cumsum,
    numpy.multiply: numpy.cumprod,
}


class FeatureNotSupported(ValueError):
    """Error for features in numpy not supported in Numpoly."""


class ndpoly(numpy.ndarray):  # pylint: disable=invalid-name
    """
    Polynomial as numpy array.

    An array object represents a multidimensional, homogeneous polynomial array
    of fixed-size items. An associated data-type object describes the format of
    each element in the array (its byte-order, how many bytes it occupies in
    memory, whether it is an integer, a floating point number, or something
    else, etc.)

    Though possible, it is not recommended to construct polynomials using
    ``ndpoly`` for basic polynomial array construction. Instead the user should
    be consider using construction functions like `symbols`, `monomial`,
    `polynomial`, etc.

    Attributes:
        coefficients (List[numpy.ndarray, ...]):
            The polynomial coefficients. Together with exponents defines the
            polynomial form.
        exponents (numpy.ndarray):
            The polynomial exponents. 2-dimensional where the first axis is the
            same length as coefficients and the second is the length of the
            indeterminant names.
        keys (List[str, ...]):
            The raw names of the coefficients. One-to-one with `exponents`, but
            as string as to be compatible with numpy structured array. Unlike
            the exponents, that are useful for mathematical manipulation, the
            keys are useful as coefficient lookup.
        indeterminants (numpoly.ndpoly):
            Secondary polynomial only consisting of an array of simple
            independent variables found in the polynomial array.
        names (Tuple[str, ...]):
            Same as `indeterminants`, but only the names as string.
        values (numpy.ndarray):
            Expose the underlying structured array.

    Examples:
        >>> poly = ndpoly(
        ...     exponents=[(0, 1), (0, 0)], shape=(3,), names="x y")
        >>> poly[";;"] = 1, 2, 3
        >>> poly[";<"] = 4, 5, 6
        >>> numpy.array(poly.coefficients)
        array([[4, 5, 6],
               [1, 2, 3]])
        >>> poly
        polynomial([4*y+1, 5*y+2, 6*y+3])
        >>> poly[0]
        polynomial(4*y+1)

    """

    # =================================================
    # Stuff to get subclassing of ndarray to run smooth
    # =================================================

    __array_priority__ = 16

    _dtype = None
    """
    The underlying structure array's actual dtype.
    Column names correspond to polynomial exponents.
    The numerical values can be calculated using the formula:
    ``poly._dtype.view(numpy.uint32)-poly.KEY_OFFSET``.
    """

    keys = None
    """
    The column names in the underlying structured array dtype
    ``ndpoly._dtype``.
    """

    names = None
    """
    Positional list of indeterminant names.
    """

    # Numpy structured array names doesn't like characters reserved by Python.
    # The largest index found with this property is 58: ':'.
    # Above this, everything looks like it works as expected.
    KEY_OFFSET = 59

    def __new__(
            cls,
            exponents=((0,),),
            shape=(),
            names=None,
            dtype=None,
            **kwargs
    ):
        """
        Class constructor.

        Args:
            exponents (numpy.ndarray):
                The exponents in an integer array with shape ``(N, D)``, where
                ``N`` is the number of terms in the polynomial sum and ``D`` is
                the number of dimensions.
            shape (Tuple[int, ...]):
                Shape of created array.
            names (Union[None, str, Tuple[str], numpoly.ndpoly]):
                The name of the indeterminant variables in the polynomial. If
                polynomial, inherent from it. Else, pass argument to
                `numpoly.symbols` to create the indeterminants names. If only
                one name is provided, but more than one is required,
                indeterminants will be extended with an integer index. If
                omitted, use ``numpoly.get_options()["default_varname"]``.
            dtype:
                Any object that can be interpreted as a numpy data type.
            kwargs:
                Extra arguments passed to `numpy.ndarray` constructor.

        """
        exponents = numpy.array(exponents, dtype=numpy.uint32)

        if names is None:
            names = option.get_options()["default_varname"]
        if isinstance(names, string_types):
            names = construct.symbols(names)
        if isinstance(names, ndpoly):
            names = names.names
        if (len(names) == 1 and not names[0][-1].isdigit() and
                (option.get_options()["force_number_suffix"] or exponents.shape[1] > 1)):
            names = tuple("%s%d" % (str(names[0]), idx)
                          for idx in range(exponents.shape[1]))
        for name in names:
            assert re.search(option.get_options()["varname_filter"], name), (
                "invalid polynomial name; "
                "expected format: %r" % option.get_options()["varname_filter"])

        if numpy.prod(exponents.shape):
            keys = (exponents+cls.KEY_OFFSET).flatten()
            keys = keys.view("U%d" % exponents.shape[-1])
        else:
            keys = numpy.zeros(0, dtype="U1")

        dtype = int if dtype is None else dtype
        dtype_ = numpy.dtype([(key, dtype) for key in keys])

        obj = super(ndpoly, cls).__new__(
            cls, shape=shape, dtype=dtype_, **kwargs)
        obj._dtype = numpy.dtype(dtype)  # pylint: disable=protected-access
        obj.keys = keys
        obj.names = tuple(names)
        return obj

    def __array_finalize__(self, obj):
        """Finalize numpy constructor."""
        if obj is None:
            return
        self.keys = getattr(obj, "keys", None)
        self.names = getattr(obj, "names", None)
        self._dtype = getattr(obj, "_dtype", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Dispatch method for operators."""
        if method == "reduce":
            ufunc = REDUCE_MAPPINGS[ufunc]
        elif method == "accumulate":
            ufunc = ACCUMULATE_MAPPINGS[ufunc]
        elif method != "__call__":
            raise FeatureNotSupported("Method '%s' not supported." % method)
        if ufunc not in dispatch.UFUNC_COLLECTION:
            raise FeatureNotSupported("ufunc '%s' not supported." % ufunc)
        return dispatch.UFUNC_COLLECTION[ufunc](*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """Dispatch method for functions."""
        if func not in dispatch.FUNCTION_COLLECTION:
            raise FeatureNotSupported(
                "function '%s' not supported by numpoly." % func.__name__)
        return dispatch.FUNCTION_COLLECTION[func](*args, **kwargs)

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
            >>> poly
            polynomial([2*x**4, -3*y**2+14])
            >>> numpy.array(poly.coefficients)
            array([[ 0, 14],
                   [ 0, -3],
                   [ 2,  0]])

        """
        if not self.size:
            return []
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
            >>> poly
            polynomial([2*x**4, -3*y**2+14])
            >>> poly.exponents
            array([[0, 0],
                   [0, 2],
                   [4, 0]], dtype=uint32)

        """
        exponents = self.keys.flatten().view(numpy.uint32)-self.KEY_OFFSET
        if numpy.prod(exponents.shape):
            exponents = exponents.reshape(len(self.keys), -1)
        assert len(exponents) > 0
        assert len(exponents.shape) == 2
        return exponents

    @staticmethod
    def from_attributes(
            exponents,
            coefficients,
            names="q",
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
            names (Union[str, Tuple[str, ...], numpoly.ndpoly]):
                The indeterminants names, either as string names or as
                simple polynomials. Must correspond to the exponents by having
                length ``D``.
            dtype (Optional[numpy.dtype]):
                The data type of the polynomial. If omitted, extract from
                `coefficients`.
            clean (bool):
                Clean up attributes, removing redundant names and exponents.
                Used to ensure alignment isn't broken.

        Returns:
            (numpoly.ndpoly):
                Polynomials with attributes defined by input.

        Examples:
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0]], coefficients=[4])
            polynomial(4)
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[1]], coefficients=[[1, 2, 3]])
            polynomial([q, 2*q, 3*q])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0], [1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q, q+1])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0, 1], [1, 1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q0*q1, q0*q1+q1])

        """
        return construct.polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=names,
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
            >>> poly
            polynomial([2*x**4, -3*y**2+14])
            >>> poly.indeterminants
            polynomial([x, y])

        """
        return construct.polynomial_from_attributes(
            exponents=numpy.eye(len(self.names), dtype=int),
            coefficients=numpy.eye(len(self.names), dtype=int),
            names=self.names,
        )

    @property
    def values(self):
        """
        Expose the underlying structured array.

        Typically used for operator dispatching and not for use to conversion.

        Examples:
            >>> numpoly.symbols("x").values
            array((1,), dtype=[('<', '<i8')])
            >>> numpoly.symbols("x y").values
            array([(1, 0), (0, 1)], dtype=[('<;', '<i8'), (';<', '<i8')])

        """
        return numpy.ndarray(
            shape=self.shape,
            dtype=[(key, self.dtype) for key in self.keys],
            buffer=self.data
        )

    def isconstant(self):
        """
        Check if a polynomial is constant or not.

        Returns:
            (bool):
                True if all elements in array are constant.

        Examples:
            >>> x = numpoly.symbols("x")
            >>> x.isconstant()
            False
            >>> numpoly.polynomial([1, 2]).isconstant()
            True
            >>> numpoly.polynomial([1, x]).isconstant()
            False

        """
        return poly_function.isconstant(self)

    def todict(self):
        """
        Cast to dict where keys are exponents and values are coefficients.

        Returns:
            (Dict[Tuple[int, ...], numpy.ndarray]):
                Dictionary where keys are exponents and values are
                coefficients.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = 2*x**4-3*y**2+14
            >>> poly
            polynomial(2*x**4-3*y**2+14)
            >>> poly.todict() == {(0, 0): 14, (4, 0): 2, (0, 2): -3}
            True

        """
        return {tuple(exponent): coefficient
                for exponent, coefficient in zip(
                    self.exponents, self.coefficients)}

    def tonumpy(self):
        """
        Cast polynomial to numpy.ndarray, if possible.

        Returns:
            (numpy.ndarray):
                Same as object, but cast to `numpy.ndarray`.

        Raises:
            numpoly.baseclass.FeatureNotSupported:
                When polynomial include indeterminats, casting to numpy.

        Examples:
            >>> numpoly.polynomial([1, 2]).tonumpy()
            array([1, 2])

        """
        return poly_function.tonumpy(self)

    # =============================================
    # Override numpy properties to work with ndpoly
    # =============================================

    @property
    def dtype(self):
        """Show coefficient dtype instead of the structured array."""
        return self._dtype

    def astype(self, dtype, **kwargs):
        """Wrap ndarray.astype."""
        coefficients = [coefficient.astype(dtype, **kwargs)
                        for coefficient in self.coefficients]
        return construct.polynomial_from_attributes(
            self.exponents, coefficients, self.names, dtype=dtype)

    def round(self, decimals=0, out=None):
        """Wrap ndarray.round."""
        # Not sure why it is required. Likely a numpy bug.
        return array_function.around(self, decimals=decimals, out=out)

    def max(self, axis=None, out=None, keepdims=False, **kwargs):
        """Wrap ndarray.max."""
        return array_function.max(self, axis=axis, out=out,
                                  keepdims=keepdims, **kwargs)

    def min(self, axis=None, out=None, keepdims=False, **kwargs):
        """Wrap ndarray.min."""
        return array_function.min(self, axis=axis, out=out,
                                  keepdims=keepdims, **kwargs)

    # ============================================================
    # Override dunder methods that isn't dealt with by dispatching
    # ============================================================

    def __call__(self, *args, **kwargs):
        """
        Evaluate polynomial by inserting new values in to the indeterminants.

        Args:
            args (int, float, numpy.ndarray, numpoly.ndpoly):
                Argument to evaluate indeterminants. Ordered positional by
                ``self.indeterminants``.
            kwargs (int, float, numpy.ndarray, numpoly.ndpoly):
                Same as ``args``, but positioned by name.

        Returns:
            (Union[numpy.ndarray, numpoly.ndpoly]):
                Evaluated polynomial. If the resulting array does not contain
                any indeterminants, an array is returned instead of a
                polynomial.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([[x, x-1], [y, y+x]])
            >>> poly(1, y=[0, 1, 2])
            array([[[1, 1, 1],
                    [0, 0, 0]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [1, 2, 3]]])

        """
        return poly_function.call(self, *args, **kwargs)

    def __eq__(self, other):
        """Left equality."""
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
            >>> poly
            polynomial([[-4*x+1, x**2],
                        [y-3, x*y**2]])
            >>> poly[0]
            polynomial([-4*x+1, x**2])
            >>> poly[:, 1]
            polynomial([x**2, x*y**2])
            >>> poly["<;"]
            array([[-4,  0],
                   [ 0,  0]])

        """
        if isinstance(index, string_types):
            return numpy.asarray(super(ndpoly, self).__getitem__(index))
        return construct.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=[coeff[index] for coeff in self.coefficients],
            names=self.names,
        )

    def __iter__(self):
        """Iterate polynomial array."""
        coefficients = numpy.array(list(self.coefficients))
        return iter(construct.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=coefficients[:, idx],
            names=self.names,
        ) for idx in range(len(self)))

    def __ne__(self, other):
        """Not equal."""
        return array_function.not_equal(self, other)

    def __repr__(self):
        """Canonical string representation."""
        return array_function.array_repr(self)

    def __str__(self):
        """Pretty string representation."""
        return array_function.array_str(self)

    def __truediv__(self, value):
        """Return self/value."""
        return poly_function.poly_divide(self, value)

    def __rtruediv__(self, value):
        """Return value/self."""
        return poly_function.poly_divide(value, self)

    def __div__(self, value):
        """Return self/value."""
        return poly_function.poly_divide(self, value)

    def __rdiv__(self, value):
        """Return value/self."""
        return poly_function.poly_divide(value, self)

    def __mod__(self, value):
        """Return self%value."""
        return poly_function.poly_remainder(self, value)

    def __rmod__(self, value):
        """Return value%self."""
        return poly_function.poly_remainder(value, self)

    def __divmod__(self, value):
        """Return divmod(self, value)."""
        return poly_function.poly_divmod(self, value)

    def __rdivmod__(self, value):
        """Return divmod(value, self)."""
        return poly_function.poly_divmod(value, self)
