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
    >>> array
    array((-1, 3, 4), dtype=[(';;', '<i8'), (';<', '<i8'), ('<;', '<i8')])

Which, together with the indeterminant names, can be used to cast back the
array back to a polynomial:

.. code:: python

    >>> numpoly.aspolynomial(array, names=("q0", "q1"))
    polynomial(3*q1+4*q0-1)
"""
from __future__ import annotations
from typing import (Any, Callable, Dict, Iterator, List,
                    Optional, Sequence, Tuple, Union)
import logging
import re

import numpy
import numpy.typing

import numpoly


REDUCE_MAPPINGS: Dict[Callable, Callable] = {
    numpy.add: numpy.sum,
    numpy.multiply: numpy.prod,
    numpy.logical_and: numpy.all,
    numpy.logical_or: numpy.any,
    numpy.maximum: numpy.amax,
    numpy.minimum: numpy.amin,
}
ACCUMULATE_MAPPINGS: Dict[Callable, Callable] = {
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

    __array_priority__: int = 16

    _dtype: numpy.dtype = numpy.dtype(int)
    """
    Underlying structure array's actual dtype.

    Column names correspond to polynomial exponents.
    The numerical values can be calculated using the formula:
    ``poly._dtype.view(numpy.uint32)-poly.KEY_OFFSET``.
    """

    keys: numpy.ndarray = numpy.empty((), dtype=int)
    """
    The raw names of the coefficients.

    One-to-one with `exponents`, but as string as to be compatible with numpy
    structured array. Unlike the exponents, that are useful for mathematical
    manipulation, the keys are useful as coefficient lookup.

    The column names in the underlying structured array dtype
    ``ndpoly._dtype``.
    """

    names: Tuple[str, ...] = ()
    """
    Same as `indeterminants`, but only the names as string.

    Positional list of indeterminant names.
    """

    allocation: int = 0
    """
    The number of polynomial coefficients allocated to polynomial.
    """

    # Numpy structured array names doesn't like characters reserved by Python.
    # The largest index found with this property is 58: ':'.
    # Above this, everything looks like it works as expected.
    KEY_OFFSET: int = 59
    """
    Internal number off-set between exponent and its stored value.

    Exponents are stored in structured array names, which are limited to not
    accept the letter ':'. By adding an offset between the represented value
    and the stored value, the letter ':' is skipped.
    """

    def __new__(
            cls,
            exponents: numpy.typing.ArrayLike = ((0,),),
            shape: Tuple[int, ...] = (),
            names: Union[None, str, Tuple[str, ...], "ndpoly"] = None,
            dtype: Optional[numpy.typing.DTypeLike] = None,
            allocation: Optional[int] = None,
            **kwargs: Any
    ) -> "ndpoly":
        """
        Class constructor.

        Args:
            exponents:
                The exponents in an integer array with shape ``(N, D)``, where
                ``N`` is the number of terms in the polynomial sum and ``D`` is
                the number of dimensions.
            shape:
                Shape of created array.
            names:
                The name of the indeterminant variables in the polynomial. If
                polynomial, inherent from it. Else, pass argument to
                `numpoly.symbols` to create the indeterminants names. If only
                one name is provided, but more than one is required,
                indeterminants will be extended with an integer index. If
                omitted, use ``numpoly.get_options()["default_varname"]``.
            dtype:
                Any object that can be interpreted as a numpy data type.
            allocation:
                The maximum number of polynomial exponents. If omitted, use
                length of exponents for allocation.
            kwargs:
                Extra arguments passed to `numpy.ndarray` constructor.

        """
        exponents = numpy.array(exponents, dtype=numpy.uint32)
        if numpy.prod(exponents.shape):
            keys = (exponents+cls.KEY_OFFSET).flatten()
            keys = keys.view(f"U{exponents.shape[-1]}")
            keys = numpy.array(keys, dtype=f"U{exponents.shape[-1]}")
        else:
            keys = numpy.full((1,), cls.KEY_OFFSET, dtype="uint32").view("U1")
        assert len(keys.shape) == 1

        dtype = int if dtype is None else dtype
        dtype_ = numpy.dtype([(key, dtype) for key in keys])

        obj = super(ndpoly, cls).__new__(
            cls, shape=shape, dtype=dtype_, **kwargs)

        if allocation is None:
            allocation = 2*len(keys)
        assert isinstance(allocation, int) and allocation >= len(keys), (
            "Not enough memory allocated; increase 'allocation'")
        if allocation > len(keys):
            allocation_ = numpy.arange(allocation-len(keys), len(keys))
            allocation_ = [str(s) for s in allocation_]
            keys = numpy.concatenate([keys, allocation_])
        obj.allocation = allocation

        if names is None:
            names = numpoly.get_options()["default_varname"]
            obj.names = numpoly.symbols(
                f"{names}:{exponents.shape[-1]}").names
        elif isinstance(names, str):
            obj.names = numpoly.symbols(names).names
        elif isinstance(names, ndpoly):
            obj.names = names.names
        else:
            obj.names = tuple(str(name) for name in names)
        for name in obj.names:
            assert re.search(numpoly.get_options()["varname_filter"], name), (
                "invalid polynomial name; "
                f"expected format: {numpoly.get_options()['varname_filter']}")

        obj._dtype = numpy.dtype(dtype)  # pylint: disable=protected-access
        obj.keys = keys
        return obj

    def __array_finalize__(self, obj: "ndpoly") -> None:
        """Finalize numpy constructor."""
        if obj is None:
            return
        self.keys = getattr(obj, "keys")
        self.names = getattr(obj, "names")
        self.allocation = getattr(obj, "allocation")
        self._dtype = getattr(obj, "_dtype")

    def __array_ufunc__(
        self,
        ufunc: Callable,
        method: Callable,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Dispatch method for operators."""
        if method == "reduce":
            ufunc = REDUCE_MAPPINGS[ufunc]
        elif method == "accumulate":
            ufunc = ACCUMULATE_MAPPINGS[ufunc]
        elif method != "__call__":
            raise FeatureNotSupported(f"Method '{method}' not supported.")
        if ufunc not in numpoly.UFUNC_COLLECTION:
            raise FeatureNotSupported(f"ufunc '{ufunc}' not supported.")
        return numpoly.UFUNC_COLLECTION[ufunc](*inputs, **kwargs)

    def __array_function__(
        self,
        func: Callable,
        types: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Dispatch method for functions."""
        del types
        logger = logging.getLogger(__name__)
        fname = func.__name__
        if func not in numpoly.FUNCTION_COLLECTION:
            raise FeatureNotSupported(
                f"function '{fname}' not supported by numpoly.")

        # notify that numpy.save* works, but numpy.load* fails
        if fname in ("save", "savez", "savez_compressed"):
            logger.warning(f"""\
numpy.{fname} used to store numpoly.ndpoly (instead of numpoly.{fname}).
This works, but restoring requires using numpoly.load, \
as numpy.load will not work as expected.""")
        elif fname == "savetxt":
            logger.warning(f"""\
numpy.{fname} used to store numpoly.ndpoly (instead of numpoly.{fname}).
This works, but restoring requires using numpoly.loadtxt, \
as numpy.loadtxt will not work as expected.""")
        return numpoly.FUNCTION_COLLECTION[func](*args, **kwargs)

    # ======================================
    # Properties specific for ndpoly objects
    # ======================================

    @property
    def coefficients(self) -> List[numpy.ndarray]:
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
    def exponents(self) -> numpy.ndarray:
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
        exponents = self.keys.astype(f"U{len(self.names)}")
        exponents = exponents.view(numpy.uint32)-self.KEY_OFFSET
        if numpy.prod(exponents.shape):
            exponents = exponents.reshape(len(self.keys), -1)
        assert len(exponents) >= 0
        assert len(exponents.shape) == 2
        return exponents

    @staticmethod
    def from_attributes(
            exponents: numpy.typing.ArrayLike,
            coefficients: Sequence[numpy.typing.ArrayLike],
            names: Union[None, str, Tuple[str, ...], "ndpoly"] = None,
            dtype: Optional[numpy.typing.DTypeLike] = None,
            allocation: Optional[int] = None,
            retain_coefficients: Optional[bool] = None,
            retain_names: Optional[bool] = None,
    ) -> "ndpoly":
        """
        Construct polynomial from polynomial attributes.

        Args:
            exponents:
                The exponents in an integer array with shape ``(N, D)``, where
                ``N`` is the number of terms in the polynomial sum and ``D`` is
                the number of dimensions.
            coefficients:
                The polynomial coefficients. Must correspond to `exponents` by
                having the same length ``N``.
            names:
                The indeterminants names, either as string names or as
                simple polynomials. Must correspond to the exponents by having
                length 1 or ``D``. If omitted, use
                ``numpoly.get_options()["default_varname"]``.
            dtype:
                The data type of the polynomial. If omitted, extract from
                `coefficients`.
            allocation:
                The maximum number of polynomial exponents. If omitted, use
                length of exponents for allocation.
            retain_coefficients:
                Do not remove redundant coefficients. If omitted use global
                defaults.
            retain_names:
                Do not remove redundant names. If omitted use global defaults.

        Returns:
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
        return numpoly.polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            names=names,
            dtype=dtype,
            allocation=allocation,
            retain_coefficients=retain_coefficients,
            retain_names=retain_names,
        )

    @property
    def indeterminants(self) -> "ndpoly":
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
        return numpoly.polynomial_from_attributes(
            exponents=numpy.eye(len(self.names), dtype=int),
            coefficients=numpy.eye(len(self.names), dtype=int),
            names=self.names,
        )

    @property
    def values(self) -> numpy.ndarray:
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

    def isconstant(self) -> bool:
        """
        Check if a polynomial is constant or not.

        Returns:
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
        return numpoly.isconstant(self)

    def todict(self) -> Dict[Tuple[int, ...], numpy.ndarray]:
        """
        Cast to dict where keys are exponents and values are coefficients.

        Returns:
            Dictionary where keys are exponents and values are coefficients.

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

    def tonumpy(self) -> numpy.ndarray:
        """
        Cast polynomial to numpy.ndarray, if possible.

        Returns:
            Same as object, but cast to `numpy.ndarray`.

        Raises:
            numpoly.baseclass.FeatureNotSupported:
                When polynomial include indeterminats, casting to numpy.

        Examples:
            >>> numpoly.polynomial([1, 2]).tonumpy()
            array([1, 2])

        """
        return numpoly.tonumpy(self)

    # =============================================
    # Override numpy properties to work with ndpoly
    # =============================================

    @property
    def dtype(self) -> numpy.dtype:
        """Datatype of the polynomial coefficients."""
        return self._dtype

    def astype(self, dtype: Any, **kwargs: Any) -> "ndpoly":  # type: ignore
        """Wrap ndarray.astype."""
        coefficients = [coefficient.astype(dtype, **kwargs)
                        for coefficient in self.coefficients]
        return numpoly.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=coefficients,
            names=self.names,
            allocation=self.allocation,
            dtype=dtype,
        )

    def diagonal(  # type: ignore
        self, offset: int = 0, axis1: int = 0, axis2: int = 1,
    ) -> "ndpoly":
        """Wrap ndarray.diagonal."""
        return numpoly.diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

    def round(  # type: ignore
        self, decimals: int = 0, out: Optional["ndpoly"] = None,
    ) -> "ndpoly":
        """Wrap ndarray.round."""
        # Not sure why it is required. Likely a numpy bug.
        return numpoly.around(self, decimals=decimals, out=out)

    def max(  # type: ignore
            self,
            axis: Optional[numpy.typing.ArrayLike] = None,
            out: Optional["ndpoly"] = None,
            keepdims: bool = False,
            **kwargs: Any,
    ) -> "ndpoly":
        """Wrap ndarray.max."""
        return numpoly.max(self, axis=axis, out=out,
                           keepdims=keepdims, **kwargs)

    def min(  # type: ignore
            self,
            axis: Optional[numpy.typing.ArrayLike] = None,
            out: Optional["ndpoly"] = None,
            keepdims: bool = False,
            **kwargs: Any,
    ) -> "ndpoly":
        """Wrap ndarray.min."""
        return numpoly.min(self, axis=axis, out=out,
                           keepdims=keepdims, **kwargs)

    def mean(  # type: ignore
        self,
        axis: Union[None, int, Sequence[int]] = None,
        dtype: Optional[numpy.typing.DTypeLike] = None,
        out: Optional[ndpoly] = None,
        **kwargs: Any,
    ) -> "ndpoly":
        """Wrap ndarray.mean."""
        return numpoly.mean(self, axis=axis, dtype=dtype, out=out, **kwargs)

    # ============================================================
    # Override dunder methods that isn't dealt with by dispatching
    # ============================================================

    def __call__(
            self,
            *args: "PolyLike",
            **kwargs: "PolyLike",
    ) -> Union[numpy.ndarray, "ndpoly"]:
        """
        Evaluate polynomial by inserting new values in to the indeterminants.

        Args:
            args:
                Argument to evaluate indeterminants. Ordered positional by
                ``self.indeterminants``.
            kwargs:
                Same as ``args``, but positioned by name.

        Returns:
            Evaluated polynomial. If the resulting array does not contain any
            indeterminants, an array is returned instead of a polynomial.

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
        return numpoly.call(self, args, kwargs)

    def __eq__(self, other: object) -> numpy.ndarray:  # type: ignore
        """Left equality."""
        return numpoly.equal(self, other)

    def __getitem__(self, index: Any) -> "ndpoly":
        """
        Get array item or slice.

        Args:
            index:
                The index to extract.

        Returns:
            Polynomial array element.

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

        """
        return numpoly.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=[coeff[index] for coeff in self.coefficients],
            names=self.names,
        )

    # def __setitem__(self, index, other):
    #     other = numpoly.aspolynomial(other)
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

    def __iter__(self) -> Iterator["ndpoly"]:
        """Iterate polynomial array."""
        coefficients = numpy.array(list(self.coefficients))
        return iter(numpoly.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=coefficients[:, idx],
            names=self.names,
        ) for idx in range(len(self)))

    def __ne__(self, other: object) -> numpy.ndarray:  # type: ignore
        """Not equal."""
        return numpoly.not_equal(self, other)

    def __repr__(self) -> str:
        """Canonical string representation."""
        return numpoly.array_repr(self)

    def __str__(self) -> str:
        """Pretty string representation."""
        return numpoly.array_str(self)

    def __truediv__(self, value: "PolyLike") -> "ndpoly":
        """Return self/value."""
        return numpoly.poly_divide(self, value)

    def __rtruediv__(self, value: "PolyLike") -> "ndpoly":
        """Return value/self."""
        return numpoly.poly_divide(value, self)

    def __div__(self, value: "PolyLike") -> "ndpoly":
        """Return self/value."""
        return numpoly.poly_divide(self, value)

    def __rdiv__(self, value: "PolyLike") -> "ndpoly":
        """Return value/self."""
        return numpoly.poly_divide(value, self)

    def __mod__(self, value: "PolyLike") -> "ndpoly":
        """Return self%value."""
        return numpoly.poly_remainder(self, value)

    def __rmod__(self, value: "PolyLike") -> "ndpoly":
        """Return value%self."""
        return numpoly.poly_remainder(value, self)

    def __divmod__(self, value: "PolyLike") -> Tuple["ndpoly", "ndpoly"]:
        """Return divmod(self, value)."""
        return numpoly.poly_divmod(self, value)

    def __rdivmod__(self, value: "PolyLike") -> Tuple["ndpoly", "ndpoly"]:
        """Return divmod(value, self)."""
        return numpoly.poly_divmod(value, self)

    def __reduce__(self) -> Tuple[Callable, Tuple]:
        """Extract state to be pickled."""
        return (numpoly.polynomial_from_attributes,
                (self.exponents, self.coefficients, self.names,
                 self.dtype, self.allocation, False))


PolyLike = Union[numpy.typing.ArrayLike, ndpoly]
