Comparison
==========

Because numbers have a natural total ordering, doing comparisons is mostly a
trivial concept. The only difficulty is how complex numbers are handled for
unsymmetrical operators. While they are not supported in pure Python:

.. code:: python

    >>> try: 1+3j > 3+1j
    ... except TypeError: print("nope")
    nope

In ``numpy``, they are supported as only comparing the real part, while
ignoring the imaginary part:

.. code:: python

    >>> (numpy.array([1+1j, 1+3j, 3+1j, 3+3j]) >
    ...  numpy.array([3+3j, 3+1j, 1+3j, 1+1j]))
    array([False, False,  True,  True])

However, polynomials comparisons are a lot more complicated as there are no
total ordering. However it is possible to impose a total order that is both
internally consistent and which is backwards compatible with the behavior of
``numpy.ndarray``. But it requires som design choices, which might have other
solutions.

The default ordering implemented in ``numpoly`` is defined as follows:

* Polynomials containing terms with the highest exponents are considered the
  largest:

  .. code:: python

    >>> q0 = numpoly.variable()
    >>> q0 < q0**2 < q0**3
    True

  If the largest polynomial in one polynomial is larger than in another,
  leading coefficients are ignored:

  .. code:: python

    >>> 4*q0 < 3*q0**2 < 2*q0**3
    True

  In the multivariate case, the polynomial order is determined by the sum of
  the exponents across the indeterminants that are multiplied together:

  .. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> q0**2*q1**2 < q0*q1**5 < q0**6*q1
    True

  This implies that given higher polynomial order, indeterminant names are
  ignored:

  .. code:: python

    >>> q0, q1, q2 = numpoly.variable(3)
    >>> q0 < q2**2 < q1**3
    True

  The same goes for any polynomial terms which are not leading:

  .. code:: python

    >>> 4*q0 < q0**2+3*q0 < q0**3+2*q0
    True


* Polynomials of equal polynomial order are sorted reverse lexicographically:

  .. code:: python

    >>> q0 < q1 < q2
    True

  As with polynomial order, coefficients and lower order terms are ignored:

  .. code:: python

    >>> 4*q0**3+4*q0 < 3*q1**3+3*q1 < 2*q2**3+2*q2
    True

  Composite polynomials of the same order are sorted by lexicographically by
  the dominant indeterminant name:

  .. code:: python

    >>> q0**3*q1 < q0**2*q1**2 < q0*q1**3
    True

  If there are more than two, and the dominant order first addresses the first,
  then the second, and so on:

  .. code:: python

    >>> q0**2*q1**2*q2 < q0**2*q1*q2**2 < q0*q1**2*q2**2
    True

* Polynomials that have exactly the same leading polynomial, are compared by
  the leading polynomial coefficient:

  .. code:: python

    >>> -4*q0 < -1*q0 < 2*q0
    True

  This notion implies that constant polynomial, the same way as ``numpy``
  arrays:

  .. code:: python

    >>> numpoly.polynomial([2, 4, 6]) > 3
    array([False,  True,  True])

* Polynomials with the same leading polynomial and coefficient are compared on
  the next largest leading polynomial:

  .. code:: python

    >>> q0**2+1 < q0**2+2 < q0**2+3
    True

  And if both the first two leading terms are the same, use the third and so
  on:

  .. code:: python

    >>> q0**2+q0+1 < q0**2+q0+2 < q0**2+q0+3
    True

  Unlike for the leading polynomial term, missing terms are considered present
  as 0. E.g.:

  .. code:: python

    >>> q0**2-1 < q0**2 < q0**2+1
    True

These rules together allow for a total comparison for all polynomials.

.. autofunction:: numpoly.poly_function.largest_exponent.largest_exponent
.. autofunction:: numpoly.poly_function.sortable_proxy.sortable_proxy

In ``numpoly``, there are a few global options that can be passed to
`numpoly.set_options` (or `numpoly.global_options`) to change this
behavior. In particular:

``sort_graded``
  Impose that polynomials are sorted by grade, meaning the indices are always
  sorted by the index sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6,
  and will therefore be consider larger than both ``q0**3*q1*q2``,
  ``q0*q1**3*q2`` and ``q0*q1*q2**3``. Defaults to true.
``sort_reverse``
  Impose that polynomials are sorted by reverses lexicographical sorting,
  meaning that ``q0*q1**3`` is considered smaller than ``q0**3*q1``, instead of the
  opposite. Defaults to false.
