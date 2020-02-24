Numpy Wrappers
==============

The numpy library comes with a large array of functions for manipulation of
`numpy.ndarray` objects. Many of these functions are supported `numpoly` as
well.

For numpy version >=1.17, the numpy library started to support dispatching
functionality to subclasses. This means that the functions in numpoly with the
same name as a numpy counterpart will work irrespectively if the function used
was from numpy or numpoly.

For example:

.. code:: python

    >>> poly = numpoly.symbols("x")**numpy.arange(4)
    >>> print(poly)
    [1 x x**2 x**3]
    >>> print(numpoly.sum(poly, keepdims=True))
    [1+x+x**2+x**3]
    >>> print(numpy.sum(poly, keepdims=True)) # doctest: +SKIP
    [1+x+x**2+x**3]

For earlier versions of numpy, the last line will not work.

Not everything is possible to support, and for the functions that are
supported, not all function are supportable.

.. automodsumm:: numpoly.array_function
   :functions-only:

.. autofunction:: numpoly.array_function.abs
.. autofunction:: numpoly.array_function.absolute
.. autofunction:: numpoly.array_function.add
.. autofunction:: numpoly.array_function.any
.. autofunction:: numpoly.array_function.all
.. autofunction:: numpoly.array_function.allclose
.. autofunction:: numpoly.array_function.around
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
.. autofunction:: numpoly.array_function.cumsum
.. autofunction:: numpoly.array_function.divide
.. autofunction:: numpoly.array_function.dsplit
.. autofunction:: numpoly.array_function.dstack
.. autofunction:: numpoly.array_function.equal
.. autofunction:: numpoly.array_function.floor
.. autofunction:: numpoly.array_function.floor_divide
.. autofunction:: numpoly.array_function.hsplit
.. autofunction:: numpoly.array_function.hstack
.. autofunction:: numpoly.array_function.inner
.. autofunction:: numpoly.array_function.isclose
.. autofunction:: numpoly.array_function.isfinite
.. autofunction:: numpoly.array_function.logical_and
.. autofunction:: numpoly.array_function.logical_or
.. autofunction:: numpoly.array_function.matmul
.. autofunction:: numpoly.array_function.mean
.. autofunction:: numpoly.array_function.moveaxis
.. autofunction:: numpoly.array_function.multiply
.. autofunction:: numpoly.array_function.negative
.. autofunction:: numpoly.array_function.not_equal
.. autofunction:: numpoly.array_function.outer
.. autofunction:: numpoly.array_function.positive
.. autofunction:: numpoly.array_function.power
.. autofunction:: numpoly.array_function.prod
.. autofunction:: numpoly.array_function.repeat
.. autofunction:: numpoly.array_function.reshape
.. autofunction:: numpoly.array_function.rint
.. autofunction:: numpoly.array_function.round
.. autofunction:: numpoly.array_function.square
.. autofunction:: numpoly.array_function.split
.. autofunction:: numpoly.array_function.stack
.. autofunction:: numpoly.array_function.subtract
.. autofunction:: numpoly.array_function.sum
.. autofunction:: numpoly.array_function.tile
.. autofunction:: numpoly.array_function.transpose
.. autofunction:: numpoly.array_function.vsplit
.. autofunction:: numpoly.array_function.vstack
