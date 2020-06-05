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

* Polynomials containing terms with the highest exponents are considered the
  largest:

  .. code:: python

    >>> x = numpoly.symbols("x")
    >>> x < x**2 < x**3
    array(True)

  If the largest polynomial in one polynomial is larger than in another,
  leading coefficients are ignored:

  .. code:: python

    >>> 4*x < 3*x**2 < 2*x**3
    array(True)

  In the multivariate case, the polynomial order is determined by the sum of
  the exponents across the indeterminants that are multiplied together:

  .. code:: python

    >>> x, y, z = numpoly.symbols("x y z")
    >>> x**2*y**2 < x*y**5 < x**6*y
    array(True)

  This implies that given higher polynomial order, indeterminant names are
  ignored:

  .. code:: python

    >>> x < z**2 < y**3
    array(True)

  The same goes for any polynomial terms which are not leading:

  .. code:: python

    >>> 4*x < x**2+3*x < x**3+2*x
    array(True)


* Polynomials of equal polynomial order are sorted reverse lexicographically:

  .. code:: python

    >>> x < y < z
    array(True)

  As with polynomial order, coefficients and lower order terms are ignored:

  .. code:: python

    >>> 4*x**3+4*x < 3*y**3+3*y < 2*z**3+2*z
    array(True)

  Composite polynomials of the same order are sorted by lexicographically by
  the dominant indeterminant name:

  .. code:: python

    >>> x**3*y < x**2*y**2 < x*y**3
    array(True)

  If there are more than two, and the dominant order first addresses the first,
  then the second, and so on:

  .. code:: python

    >>> x**2*y**2*z < x**2*y*z***2 < x*y**2*z**2
    array(True)

* Polynomials that have exactly the same leading polynomial, are compared by
  the leading polynomial coefficient:

  .. code:: python

    >>> -4*x < -1*x < 2*x
    array(True)

  This notion implies that constant polynomial, the same way as ``numpy``
  arrays:

  .. code:: python

    >>> numpoly.polynomial([2, 4, 6]) > 3
    array([False,  True,  True])

  It is worth noting that all non-constant polynomials will be larger than the
  constant 0, including negative ones. This gives break the following intuitive
  behavior:

  .. code:: python

    >>> -4*x < 0*x < 4*x
    array(False)

  Instead, the following holds true:

  .. code:: python

    >>> 0*x == 0
    array(True)
    >>> 0*x < -4*x
    array(True)

* Polynomials with the same leading polynomial and coefficient are compared on
  the next largest leading polynomial:

  .. code:: python

    >>> x**2+1 < x**2+2 < x**2+3
    array(True)

  And if both the first two leading terms are the same, use the third and so
  on:

  .. code:: python

    >>> x**2+x+1 < x**2+x+2 < x**2+x+3
    array(True)

  Unlike for the leading polynomial term, missing terms are considered present
  as 0. E.g.:

  .. code:: python

    >>> x**2-1 < x**2 < x**2+1
    array(True)

These rules together allow for a total comparison for all polynomials.

.. autofunction:: numpoly.poly_function.sortable_proxy.sortable_proxy
