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

However, for polynomials comparisons are a lot more complicated as there are no
total ordering. However it is possible to impose a total order that is both
internally consistent and which is backwards compatible with the behavior of
``numpy.ndarray``. The ordering implemented in ``numpoly`` is defined as
follows:

* Constants compare as in ``numpy``:

  .. code:: python

    >>> numpoly.polynomial([2, 4, 6]) > 3
    array([False,  True,  True])

* Indeterminants are sorted reversed lexicographically:

  .. code:: python

    >>> x, y, z = numpoly.symbols("x y z")
    >>> x < y < z
    array(True)

* Simple polynomials with composite indeterminant and exponents are sorted
  using graded reverse graded lexicographically:

  .. code:: python

    >>> x**2 < x**3 < x**4
    array(True)
    >>> x**3 < x**2*y < x*y**2
    array(True)

* Simple polynomials with matching indeterminants and exponents are compared on
  coefficients:

  .. code:: python

    >>> numpoly.polynomial([2*x, 4*x, 6*x]) > 3*x
    array([False,  True,  True])

* Polynomials with multiple terms will be foremost be compared by the
  coefficients to the indeterminants with the largest exponents:

  .. code:: python

    >>> numpoly.polynomial([2*x+5, 4*x-1, 6*x+1]) > 3*x+10
    array([False,  True,  True])

* If the lead coefficients is the same, the polynomial is compared on the
  second lead. If those match, go to the third and so on. E.g.

  .. code:: python

    >>> numpoly.polynomial([2*x**2, x**2-x, x**2+x+3]) > x**2+x+1
    array([ True, False,  True])

* Except the lead, missing coefficient are assumed to be zero:

  .. code:: python

    >>> numpoly.polynomial([x**2-x, x**2+x]) > x**2
    array([False,  True])

These rules together allow for a total comparison for all polynomials.

.. autofunction:: numpoly.poly_function.sortable_proxy.sortable_proxy
