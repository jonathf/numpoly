Introduction
============

In `numpy`_ the concept of *array* is generalized to imply arrays of arbitrary
dimension, overlapping with the concept of *scalars*, *matrices* and *tensors*.
To allow arrays of various dimensions to operate together it defines
unambiguous broadcasting rules for what to expect. The results is a library
that is used as the reference for almost all of the numerical Python community.

In mathematical literature the term *polynomial expansions* is used to denote a
collection of polynomials. Though they strictly do not need to, they are often
indexed, giving each polynomial both a label and a position for where to locate
a polynomial relative to the others. Assuming that there always is an index,
one could say that *polynomial expansions* could just as well be termed
*polynomial array*. And using the rules defined in `numpy`_, there no reason
not to also start talking about *multi-dimensional polynomial arrays*.

The main idea here is that in the same way as :class:`numpy.ndarray` are
composed of scalars, :class:`numpoly.ndpoly` -- the baseclass for the
polynomial arrays -- are composed of simpler polynomials. This gives us a
mental model of a polynomial that looks like this:

.. math::

    \Phi(q_1, \dots, q_D) =
        [\Phi_1(q_1, \dots, q_D), \cdots, \Phi_N(q_1, \dots, q_D)]

where :math:`\Phi` is polynomial vector, :math:`N` is the number of terms in
the polynomial sum, and :math:`q_d` is the :math:`d`-th indeterminant name.
This mental model is shown in practice in how numpoly displays its polynomials
in the REPL:

.. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> expansion = numpoly.polynomial([1, q0, q1**2])
    >>> expansion
    polynomial([1, q0, q1**2])

Another way to look at the polynomials is to keep the polynomial array as a
single polynomial sum: A multivariate polynomial can in the case of ``numpoly``
be defined as:

.. math::

    \Phi(q_1, \dots, q_D) = \sum_{n=1}^N c_n q_1^{k_{1n}} \cdots q_D^{k_{Dn}}

where :math:`c_n` is a multi-dimensional polynomial
coefficients, and :math:`k_{nd}` is the exponent for the :math:`n`-th
polynomial term and the :math:`d`-th indeterminant name.

Neither of the two ways of representing a polynomial array is incorrect, and
serves different purposes. The former works well for visualisation, while the
latter form gives a better mental model of how ``numpoly`` handles its
polynomial internally.

Modelling polynomials by storing the coefficients as multi-dimensional arrays
is deliberate. Assuming few :math:`k_{nd}` and large dimensional :math:`c_n`,
all numerical operations that are limited to the coefficients, can be done
fast, as `numpy`_ can do the heavy lifting.

This way of representing a polynomial also means that to uniquely defined a
polynomial, we only need the three components:

* ``coefficients`` -- the polynomial coefficients :math:`c_n` as
  multi-dimensional arrays.
* ``exponents`` -- the exponents :math:`k_{nd}` as a 2-dimensional matrix.
* ``indeterminants`` -- the names of the variables, typically ``q0``, ``q1``,
  etc.

We can access these three defining properties directly from any ``numpoly``
polynomials. For example, for a simple polynomial with scalar coefficients:

.. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> poly = numpoly.polynomial(4*q0+3*q1-1)
    >>> poly
    polynomial(3*q1+4*q0-1)
    >>> indet = poly.indeterminants
    >>> indet
    polynomial([q0, q1])
    >>> coeff = poly.coefficients
    >>> coeff
    [-1, 4, 3]
    >>> expon = poly.exponents
    >>> expon
    array([[0, 0],
           [1, 0],
           [0, 1]], dtype=uint32)

Because these three properties uniquely define a polynomial array, they can
also be used to reconstruct the original polynomial:

.. code:: python

    >>> terms = coeff*numpoly.prod(indet**expon, -1)
    >>> terms
    polynomial([-1, 4*q0, 3*q1])
    >>> poly = numpoly.sum(terms, axis=0)
    >>> poly
    polynomial(3*q1+4*q0-1)

.. note::

    As mentioned the chosen representation works best with relatively few
    :math:`k_{nd}` and large :math:`c_n`. for large number :math:`k_{nd}` and
    relatively small :math:`c_n` however, the advantage disappears. And even
    worse, in the case where polynomial terms :math:`q_1^{k_{1n}} \cdots
    q_D^{k_{Dn}}` are sparsely represented, the ``numpoly`` representation is
    quite memory inefficient. So it is worth keeping in mind that the advantage
    of this implementation depends a little upon what kind of problems you are
    working on. It is not the tool for all problems.

.. _numpy: https://numpy.org/doc/stable

Numpy functions
===============

The ``numpoly`` concept of arrays is taken from `numpy`_. But it goes a bit deeper
than just inspiration. The base class
:class:`numpoly.ndpoly` is a direct subclass of
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

    >>> q0, q1 = numpoly.variable(2)
    >>> poly_array = numpoly.polynomial([[1, q0-1], [q1**2, 4]])
    >>> numpy.transpose(poly_array)
    polynomial([[1, q1**2],
                [q0-1, 4]])

Though the overlap in functionality between `numpy`_ and ``numpoly`` is large,
there are still lots of functionality which is specific for each of them.
The most obvious, in the case of ``numpoly`` features not found in `numpy`_ is
the ability to evaluate the polynomials:

.. code:: python

    >>> poly = q1**2-q0
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

Comparison operators
====================

Because numbers have a natural total ordering, doing comparisons is mostly a
trivial concept. The only difficulty is how complex numbers are handled for
unsymmetrical operators. While they are not supported in pure Python:

.. code:: python

    >>> from contextlib import suppress
    >>> with suppress(TypeError):
    ...     1+3j > 3+1j

In ``numpy``, comparisons are supported, but limited to the real part,
ignoring the imaginary part:

.. code:: python

    >>> (numpy.array([1+1j, 1+3j, 3+1j, 3+3j]) >
    ...  numpy.array([3+3j, 3+1j, 1+3j, 1+1j]))
    array([False, False,  True,  True])

Polynomials comparisons are a lot more complicated as there are no total
ordering. However, it is possible to impose a total order that is both
internally consistent and which is backwards compatible with the behavior of
``numpy.ndarray``. It requires som design choices, which is opinionated, and
might not always align with everyones taste.

The default ordering implemented in ``numpoly`` is defined as follows:

* Polynomials containing terms with the highest exponents are considered the
  largest:

  .. code:: python

    >>> q0 = numpoly.variable()
    >>> q0 < q0**2 < q0**3
    True

  If the largest polynomial exponent in one polynomial is larger than in
  another, leading coefficients are ignored:

  .. code:: python

    >>> 4*q0 < 3*q0**2 < 2*q0**3
    True

  In the multivariate case, the polynomial order is determined by the sum of
  the exponents across the indeterminants that are multiplied together:

  .. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> q0**2*q1**2 < q0*q1**5 < q0**6*q1
    True

  This implies that given a higher polynomial order, indeterminant names are
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

  As with polynomial order, coefficients and lower order terms are also
  ignored:

  .. code:: python

    >>> 4*q0**3+4*q0 < 3*q1**3+3*q1 < 2*q2**3+2*q2
    True

  Composite polynomials of the same order are sorted by lexicographically by
  the dominant indeterminant name:

  .. code:: python

    >>> q0**3*q1 < q0**2*q1**2 < q0*q1**3
    True

  If there are more than two indeterminants, the dominant order first
  addresses the first name (sorted lexicographically), then the second, and so
  on:

  .. code:: python

    >>> q0**2*q1**2*q2 < q0**2*q1*q2**2 < q0*q1**2*q2**2
    True

* Polynomials that have the same leading polynomial exponents, are compared by
  the leading polynomial coefficient:

  .. code:: python

    >>> -4*q0 < -1*q0 < 2*q0
    True

  This notion implies that constant polynomials behave in the same way as
  ``numpy`` arrays:

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

In ``numpoly``, there are a few global options that can be passed to
:func:`numpoly.set_options` (or :func:`numpoly.global_options`) to change this
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

Polynomial division
===================

Numerical division can be split into two variants: floor division and true
division:

.. code:: python

    >>> dividend = 7
    >>> divisor = 2
    >>> quotient_true = numpy.true_divide(dividend, divisor)
    >>> quotient_true
    3.5
    >>> quotient_floor = numpy.floor_divide(dividend, divisor)
    >>> quotient_floor
    3

The discrepancy between the two can be captured by a remainder, which allow us
to more formally define them as follows:

.. code:: python

    >>> remainder = numpy.remainder(dividend, divisor)
    >>> remainder
    1
    >>> dividend == quotient_floor*divisor+remainder
    True
    >>> dividend == quotient_true*divisor
    True


In the case of polynomials, neither true nor floor division is supported like
this. Instead it support its own kind of polynomial division. Polynomial
division falls back to behave like floor division for all constants, as it does
not round values:

.. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> dividend = q0**2+q1
    >>> divisor = q0-1
    >>> quotient = numpoly.poly_divide(dividend, divisor)
    >>> quotient
    polynomial(q0+1.0)

However, like floor division, it can still have remainders.
For example:

.. code:: python

    >>> remainder = numpoly.poly_remainder(dividend, divisor)
    >>> remainder
    polynomial(q1+1.0)
    >>> dividend == quotient*divisor+remainder
    True

In ``numpy``, the "Python syntactic sugar" operators have the following
behavior:

* ``/`` is used for true division.
* ``//`` is used for floor division.
* ``%`` is used for remainder.
* ``divmod`` is used for floor division and remainder in combination to save
  computational cost.

In ``numpoly``, which takes precedence if any of the values are of
``numpoly.ndpoly`` objects, take the following behavior:

* ``/`` is used for polynomial division, which is backwards compatible with
  ``numpy``.
* ``//`` is still used for floor division as in ``numpy``, which is only
  possible if divisor is a constant.
* ``%`` is used for polynomial remainder, which is not backwards compatible.
* ``divmod`` is used for polynomial division and remainder in combination to
  save computation cost.