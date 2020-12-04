"""
Polynomial base class.

Under the hood, each of the polynomials are numpy structured arrays. The column
names are string representations corresponding to the polynomial exponents, and
the values are the coefficients. The indeterminant names are stored separately.
Numpoly is a wrapper on top of this.

From a development point of view, it is possible to view the underlying
structured array directly using the ``values`` attribute:

.. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> poly = numpoly.polynomial(4*q0+3*q1-1)
    >>> array = poly.values
    >>> array  # doctest: +NORMALIZE_WHITESPACE
    array((-1, 4, 3),
          dtype=[(';;', '<i8'), ('<;', '<i8'), (';<', '<i8')])

Which, together with the indeterminant names, can be used to cast back the
array back to a polynomial:

.. code:: python

    >>> numpoly.aspolynomial(array, names=("q0", "q1"))
    polynomial(3*q1+4*q0-1)
"""
from __future__ import division
import logging
import re
from six import string_types

import numpy
from numpy.lib import recfunctions

from . import align, construct, dispatch, array_function, poly_function, option


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
    be consider using construction functions like `variable`, `monomial`,
    `polynomial`, etc.

    Examples:
        >>> poly = ndpoly(
        ...     exponents=[(0, 1), (0, 0)], shape=(3,))
        >>> poly.values[";<"] = 4, 5, 6
        >>> poly.values[";;"] = 1, 2, 3
        >>> numpy.array(poly.coefficients)
        array([[4, 5, 6],
               [1, 2, 3]])
        >>> poly
        polynomial([4*q1+1, 5*q1+2, 6*q1+3])
        >>> poly[0]
        polynomial(4*q1+1)

    """

    # =================================================
    # Stuff to get subclassing of ndarray to run smooth
    # =================================================

    __array_priority__ = 16

    _dtype = None
    """
    Underlying structure array's actual dtype.

    Column names correspond to polynomial exponents.
    The numerical values can be calculated using the formula:
    ``poly._dtype.view(numpy.uint32)-poly.KEY_OFFSET``.
    """

    keys = None
    """
    The raw names of the coefficients.

    One-to-one with `exponents`, but as string as to be compatible with numpy
    structured array. Unlike the exponents, that are useful for mathematical
    manipulation, the keys are useful as coefficient lookup.

    The column names in the underlying structured array dtype
    ``ndpoly._dtype``.
    """

    names = None
    """
    Same as `indeterminants`, but only the names as string.

    Positional list of indeterminant names.
    """

    allocation = None
    """
    The number of polynomial coefficients allocated to polynomial.
    """

    # Numpy structured array names doesn't like characters reserved by Python.
    # The largest index found with this property is 58: ':'.
    # Above this, everything looks like it works as expected.
    KEY_OFFSET = 59
    """
    Internal number off-set between exponent and its stored value.

    Exponents are stored in structured array names, which are limited to not accept the letter ':'. By adding an offset between the represented value and the stored value, the letter ':' is skipped.
    """

    def __new__(
            cls,
            exponents=((0,),),
            shape=(),
            names=None,
            dtype=None,
            allocation=None,
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
            dtype (Optional[numpy.dtype]):
                Any object that can be interpreted as a numpy data type.
            allocation (Optional[int]):
                The maximum number of polynomial exponents. If omitted, use
                length of exponents for allocation.
            kwargs:
                Extra arguments passed to `numpy.ndarray` constructor.

        """
        exponents = numpy.array(exponents, dtype=numpy.uint32)

        if numpy.prod(exponents.shape):
            keys = (exponents+cls.KEY_OFFSET).flatten()
            keys = keys.view("U%d" % exponents.shape[-1])
            keys = numpy.array(keys, dtype="U%d" % (exponents.shape[-1]))
        else:
            keys = numpy.zeros(0, dtype="U1")

        if allocation is None:
            allocation = 2*len(keys)
        assert isinstance(allocation, int) and allocation >= len(keys), (
            "Not enough memory allocated; increase 'allocation'")
        if allocation > len(keys):
            allocation_ = numpy.arange(allocation-len(keys), len(keys))
            allocation_ = [str(s) for s in allocation_]
            keys = numpy.concatenate([keys, allocation_])

        if names is None:
            names = option.get_options()["default_varname"]
            names = construct.symbols("%s:%d" % (names, exponents.shape[-1])).names
        elif isinstance(names, string_types):
            names = construct.symbols(names).names
        elif isinstance(names, ndpoly):
            names = names.names
        for name in names:
            assert re.search(option.get_options()["varname_filter"], name), (
                "invalid polynomial name; "
                "expected format: %r" % option.get_options()["varname_filter"])

        dtype = int if dtype is None else dtype
        dtype_ = numpy.dtype([(key, dtype) for key in keys])

        obj = super(ndpoly, cls).__new__(
            cls, shape=shape, dtype=dtype_, **kwargs)
        obj._dtype = numpy.dtype(dtype)  # pylint: disable=protected-access
        obj.keys = keys
        obj.names = tuple(names)
        obj.allocation = allocation
        return obj

    def __array_finalize__(self, obj):
        """Finalize numpy constructor."""
        if obj is None:
            return
        self.keys = getattr(obj, "keys", None)
        self.names = getattr(obj, "names", None)
        self.allocation = getattr(obj, "allocation", None)
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
        logger = logging.getLogger(__name__)
        fname = func.__name__
        if func not in dispatch.FUNCTION_COLLECTION:
            raise FeatureNotSupported(
                "function '%s' not supported by numpoly." % fname)

        # notify that numpy.save* works, but numpy.load* fails
        if fname in ("save", "savez", "savez_compressed"):
            logger.warning("""\
numpy.%s used to store numpoly.ndpoly (instead of numpoly.%s).
This works, but restoring requires using numpoly.load, \
as numpy.load will not work as expected.""" % (fname, fname))
        elif fname == "savetxt":
            logger.warning("""\
numpy.%s used to store numpoly.ndpoly (instead of numpoly.%s).
This works, but restoring requires using numpoly.loadtxt, \
as numpy.loadtxt will not work as expected.""" % (fname, fname))
        return dispatch.FUNCTION_COLLECTION[func](*args, **kwargs)

    # ======================================
    # Properties specific for ndpoly objects
    # ======================================

    @property
    def coefficients(self):
        """
        Polynomial coefficients.

        Together with exponents defines the polynomial form.

        Examples:
            >>> q0, q1 = numpoly.variable(2)
            >>> poly = numpoly.polynomial([2*q0**4, -3*q1**2+14])
            >>> poly
            polynomial([2*q0**4, -3*q1**2+14])
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

        2-dimensional where the first axis is the same length as coefficients
        and the second is the length of the indeterminant names.

        Examples:
            >>> q0, q1 = numpoly.variable(2)
            >>> poly = numpoly.polynomial([2*q0**4, -3*q1**2+14])
            >>> poly
            polynomial([2*q0**4, -3*q1**2+14])
            >>> poly.exponents
            array([[0, 0],
                   [0, 2],
                   [4, 0]], dtype=uint32)

        """
        exponents = self.keys.astype("U%d" % len(self.names))
        exponents = exponents.view(numpy.uint32)-self.KEY_OFFSET
        if numpy.prod(exponents.shape):
            exponents = exponents.reshape(len(self.keys), -1)
        assert len(exponents) >= 0
        assert len(exponents.shape) == 2
        return exponents

    @staticmethod
    def from_attributes(
            exponents,
            coefficients,
            names=None,
            dtype=None,
            allocation=None,
            retain_coefficients=None,
            retain_names=None,
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
            names (Union[None, str, Tuple[str, ...], numpoly.ndpoly]):
                The indeterminants names, either as string names or as
                simple polynomials. Must correspond to the exponents by having
                length 1 or ``D``. If omitted, use
                ``numpoly.get_options()["default_varname"]``.
            dtype (Optional[numpy.dtype]):
                The data type of the polynomial. If omitted, extract from
                `coefficients`.
            allocation (Optional[int]):
                The maximum number of polynomial exponents. If omitted, use
                length of exponents for allocation.
            retain_coefficients (Optional[bool]):
                Do not remove redundant coefficients. If omitted use global
                defaults.
            retain_names (Optional[bool]):
                Do not remove redundant names. If omitted use global defaults.

        Returns:
            (numpoly.ndpoly):
                Polynomials with attributes defined by input.

        Examples:
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0]], coefficients=[4])
            polynomial(4)
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[1]], coefficients=[[1, 2, 3]])
            polynomial([q0, 2*q0, 3*q0])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0], [1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q0, q0+1])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0, 1], [1, 1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q0*q1, q0*q1+q1])

        """
        return construct.polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=names,
            dtype=dtype,
            allocation=allocation,
            retain_coefficients=retain_coefficients,
            retain_names=retain_names,
        )

    @property
    def indeterminants(self):
        """
        Polynomial indeterminants.

        Secondary polynomial only consisting of an array of simple independent
        variables found in the polynomial array.

        Examples:
            >>> q0, q1 = numpoly.variable(2)
            >>> poly = numpoly.polynomial([2*q0**4, -3*q1**2+14])
            >>> poly
            polynomial([2*q0**4, -3*q1**2+14])
            >>> poly.indeterminants
            polynomial([q0, q1])

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
            >>> numpoly.variable(1).values
            array((1,), dtype=[('<', '<i8')])
            >>> numpoly.variable(2).values
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
            >>> q0 = numpoly.variable()
            >>> q0.isconstant()
            False
            >>> numpoly.polynomial([1, 2]).isconstant()
            True
            >>> numpoly.polynomial([1, q0]).isconstant()
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
            >>> q0, q1 = numpoly.variable(2)
            >>> poly = 2*q0**4-3*q1**2+14
            >>> poly
            polynomial(2*q0**4-3*q1**2+14)
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
        """Datatype of the polynomial coefficients."""
        return self._dtype

    def astype(self, dtype, **kwargs):
        """Wrap ndarray.astype."""
        coefficients = [coefficient.astype(dtype, **kwargs)
                        for coefficient in self.coefficients]
        return construct.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=coefficients,
            names=self.names,
            allocation=self.allocation,
            dtype=dtype,
        )

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Wrap ndarray.diagonal."""
        return array_function.diagonal(
            self, offset=offset, axis1=axis1, axis2=axis2)

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
            >>> q0, q1 = numpoly.variable(2)
            >>> poly = numpoly.polynomial(
            ...     [[q0, q0-1], [q1, q1+q0]])
            >>> poly(1, q1=[0, 1, 2])
            array([[[1, 1, 1],
                    [0, 0, 0]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [1, 2, 3]]])

        """
        return poly_function.call(self, args, kwargs)

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
            >>> q0, q1 = numpoly.variable(2)
            >>> poly = numpoly.polynomial([[1-4*q0, q0**2], [q1-3, q0*q1*q1]])
            >>> poly
            polynomial([[-4*q0+1, q0**2],
                        [q1-3, q0*q1**2]])
            >>> poly[0]
            polynomial([-4*q0+1, q0**2])
            >>> poly[:, 1]
            polynomial([q0**2, q0*q1**2])
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

    # def __setitem__(self, index, other):
    #     other = construct.aspolynomial(other)
    #     difference =  set(other.names).difference(self.names)
    #     assert not difference, (
    #         "polynomial does not contain indeterminants: %s" % difference)
    #     assert self.ndim >= other.ndim
    #     other, _ = align.align_polynomials(other, self)
    #     for key in other.keys:
    #         if key not in self.keys:
    #             for idx, key_ in enumerate(self.values.dtype.names):
    #                 if key_.isdigit():
    #                     break
    #             else:
    #                 fail
    #             names = list(self.values.dtype.names)
    #             names[idx] = key
    #         self.values[key][index] = other.values[key][index]


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

    def __reduce__(self):
        """Extract state to be pickled."""
        return (construct.polynomial_from_attributes,
                (self.exponents, self.coefficients, self.names,
                 self.dtype, self.allocation, False))
