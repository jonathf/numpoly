Numpy
=====

The ``numpoly`` concept of arrays is taken from `numpy`_. But it goes a bit deeper
than just inspiration. The base class
:class:`numpoly.ndpoly <numpoly.baseclass.ndpoly>` is a direct subclass of
:class:`numpy.ndarray`:

.. code:: python

    >>> issubclass(numpoly.ndpoly, numpy.ndarray)
    True

The intentions is to have a library that is fast with the respect of the number
of coefficients, as it leverages `numpy`_'s speed where possible.

In addition ``numpoly`` is designed to be behave both as you would expect as a
polynomial, but also, where possible, to behave as a `numpy`_ numerical array.
In practice this means that ``numpoly`` provides a lot functions that also
exists in `numpy`_, which does about the same thing. If one of these
``numpoly`` function is provided with a :class:`numpy.ndarray` object, the
returned values is the same as if provided to the `numpy`_ function with the
same name. For example:

.. code:: python

    >>> num_array = numpy.array([[1, 2], [3, 4]])
    >>> numpoly.transpose(num_array)
    polynomial([[1, 3],
                [2, 4]])

And this works the other way around as well. If a polynomial is provided to the
`numpy`_ function, it will behave the same way as if it was provided to the
``numpoly`` equivalent. For example:

.. code:: python

    >>> poly_array = numpoly.polynomial([[1, q0-1], [q1**2, 4]])
    >>> numpy.transpose(poly_array)
    polynomial([[1, q1**2],
                [q0-1, 4]])

Though the overlap in functionality between `numpy`_ and ``numpoly`` is large,
there are still lots of functionality which is specific for each of them.
The most obvious, in the case of ``numpoly`` feature not found in `numpy`_ is
the ability to evaluate the polynomials:

.. code:: python

    >>> poly
    polynomial(q1**2-q0)
    >>> poly(4, 4)
    12
    >>> poly(4)
    polynomial(q1**2-4)
    >>> poly([1, 2, 3])
    polynomial([q1**2-1, q1**2-2, q1**2-3])

Function Compatibility
----------------------

The numpy library comes with a large number of functions for manipulating
:class:`numpy.ndarray` objects. Many of these functions are supported
``numpoly`` as well.

For numpy version >=1.17, the `numpy`_ library introduced dispatching of its
functions to subclasses. This means that functions in ``numpoly`` with the
same name as a numpy counterpart, it will work the same irrespectively if the
function used was from `numpy`_ or ``numpoly``, as the former will pass any
job to the latter.

For example:

.. code:: python

    >>> poly = numpoly.variable()**numpy.arange(4)
    >>> print(poly)
    [1 q0 q0**2 q0**3]
    >>> print(numpoly.sum(poly, keepdims=True))
    [q0**3+q0**2+q0+1]
    >>> print(numpy.sum(poly, keepdims=True)) # doctest: +SKIP
    [q0**3+q0**2+q0+1]

For earlier versions of numpy, the last line will not work.

Not everything is possible to support, and even within the list of supported
functions, not all use cases can be covered. Bit if such an unsupported edge
case is encountered, an ``numpoly.baseclass.FeatureNotSupported`` error should
be raised, so it should be obvious when they happen.

As a developer note, ``numpoly`` aims at being backwards compatible with
`numpy`_ as far as possible when it comes to the functions. This means that all
functions below should as far as possible mirror the behavior their `numpy`_
counterparts, and for polynomial constant, they should be identical (except for
the object type). Function that provides behavior not covered by `numpy`_
should be placed elsewhere.

.. _numpy: https://numpy.org/doc/stable

Collection
----------

.. automodsumm:: numpoly.array_function
   :functions-only:

.. autofunction:: numpoly.array_function.abs
.. autofunction:: numpoly.array_function.absolute
.. autofunction:: numpoly.array_function.add
.. autofunction:: numpoly.array_function.any
.. autofunction:: numpoly.array_function.all
.. autofunction:: numpoly.array_function.allclose
.. autofunction:: numpoly.array_function.amax
.. autofunction:: numpoly.array_function.amin
.. autofunction:: numpoly.array_function.argmin
.. autofunction:: numpoly.array_function.argmax
.. autofunction:: numpoly.array_function.around
.. autofunction:: numpoly.array_function.apply_along_axis
.. autofunction:: numpoly.array_function.apply_over_axes
.. autofunction:: numpoly.array_function.array_repr
.. autofunction:: numpoly.array_function.array_split
.. autofunction:: numpoly.array_function.array_str
.. autofunction:: numpoly.array_function.atleast_1d
.. autofunction:: numpoly.array_function.atleast_2d
.. autofunction:: numpoly.array_function.atleast_3d
.. autofunction:: numpoly.array_function.broadcast_arrays
.. autofunction:: numpoly.array_function.ceil
.. autofunction:: numpoly.array_function.choose
.. autofunction:: numpoly.array_function.common_type
.. autofunction:: numpoly.array_function.concatenate
.. autofunction:: numpoly.array_function.copyto
.. autofunction:: numpoly.array_function.count_nonzero
.. autofunction:: numpoly.array_function.cumsum
.. autofunction:: numpoly.array_function.diag
.. autofunction:: numpoly.array_function.diagonal
.. autofunction:: numpoly.array_function.diff
.. autofunction:: numpoly.array_function.divide
.. autofunction:: numpoly.array_function.divmod
.. autofunction:: numpoly.array_function.dsplit
.. autofunction:: numpoly.array_function.dstack
.. autofunction:: numpoly.array_function.ediff1d
.. autofunction:: numpoly.array_function.equal
.. autofunction:: numpoly.array_function.expand_dims
.. autofunction:: numpoly.array_function.floor
.. autofunction:: numpoly.array_function.floor_divide
.. autofunction:: numpoly.array_function.full
.. autofunction:: numpoly.array_function.full_like
.. autofunction:: numpoly.array_function.greater
.. autofunction:: numpoly.array_function.greater_equal
.. autofunction:: numpoly.array_function.hsplit
.. autofunction:: numpoly.array_function.hstack
.. autofunction:: numpoly.array_function.inner
.. autofunction:: numpoly.array_function.isclose
.. autofunction:: numpoly.array_function.isfinite
.. autofunction:: numpoly.array_function.less
.. autofunction:: numpoly.array_function.less_equal
.. autofunction:: numpoly.array_function.load
.. autofunction:: numpoly.array_function.loadtxt
.. autofunction:: numpoly.array_function.logical_and
.. autofunction:: numpoly.array_function.logical_or
.. autofunction:: numpoly.array_function.matmul
.. autofunction:: numpoly.array_function.mean
.. autofunction:: numpoly.array_function.max
.. autofunction:: numpoly.array_function.maximum
.. autofunction:: numpoly.array_function.min
.. autofunction:: numpoly.array_function.minimum
.. autofunction:: numpoly.array_function.moveaxis
.. autofunction:: numpoly.array_function.mod
.. autofunction:: numpoly.array_function.multiply
.. autofunction:: numpoly.array_function.negative
.. autofunction:: numpoly.array_function.nonzero
.. autofunction:: numpoly.array_function.not_equal
.. autofunction:: numpoly.array_function.ones
.. autofunction:: numpoly.array_function.ones_like
.. autofunction:: numpoly.array_function.outer
.. autofunction:: numpoly.array_function.positive
.. autofunction:: numpoly.array_function.power
.. autofunction:: numpoly.array_function.prod
.. autofunction:: numpoly.array_function.remainder
.. autofunction:: numpoly.array_function.repeat
.. autofunction:: numpoly.array_function.reshape
.. autofunction:: numpoly.array_function.result_type
.. autofunction:: numpoly.array_function.rint
.. autofunction:: numpoly.array_function.round
.. autofunction:: numpoly.array_function.round_
.. autofunction:: numpoly.array_function.save
.. autofunction:: numpoly.array_function.savetxt
.. autofunction:: numpoly.array_function.savez
.. autofunction:: numpoly.array_function.savez_compressed
.. autofunction:: numpoly.array_function.square
.. autofunction:: numpoly.array_function.split
.. autofunction:: numpoly.array_function.stack
.. autofunction:: numpoly.array_function.subtract
.. autofunction:: numpoly.array_function.sum
.. autofunction:: numpoly.array_function.tile
.. autofunction:: numpoly.array_function.transpose
.. autofunction:: numpoly.array_function.true_divide
.. autofunction:: numpoly.array_function.vsplit
.. autofunction:: numpoly.array_function.vstack
.. autofunction:: numpoly.array_function.where
.. autofunction:: numpoly.array_function.zeros
.. autofunction:: numpoly.array_function.zeros_like
