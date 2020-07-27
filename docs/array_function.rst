Numpy Wrappers
==============

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
.. autofunction:: numpoly.array_function.count_nonzero
.. autofunction:: numpoly.array_function.cumsum
.. autofunction:: numpoly.array_function.diag
.. autofunction:: numpoly.array_function.diagonal
.. autofunction:: numpoly.array_function.diff
.. autofunction:: numpoly.array_function.divide
.. autofunction:: numpoly.array_function.divmod
.. autofunction:: numpoly.array_function.dsplit
.. autofunction:: numpoly.array_function.dstack
.. autofunction:: numpoly.array_function.equal
.. autofunction:: numpoly.array_function.expand_dims
.. autofunction:: numpoly.array_function.floor
.. autofunction:: numpoly.array_function.floor_divide
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
