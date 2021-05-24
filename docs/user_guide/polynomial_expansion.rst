Polynomial expansions
---------------------

A simple polynomial can be created through variable constructor
:func:`numpoly.variable`. For example to construct a simple bivariate
polynomial:

.. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> q0
    polynomial(q0)

A collection of polynomial can be manipulated using basic arithmetic operators
and joined together into polynomial expansions:

.. code:: python

    >>> poly = numpoly.polynomial([1, q0, 1-q0*q1, q0**2*q1, q0-q1**2])
    >>> poly
    polynomial([1, q0, -q0*q1+1, q0**2*q1, -q1**2+q0])

Note that constants and simple polynomials can be joined together into arrays
without any problems.

In practice, having the ability to fine tune a polynomial exactly as one wants
it can be useful, but it can also be cumbersome when dealing with larger arrays
for application. To automate the construction of simple polynomials, there is
the :func:`numpoly.monomial` constructor. In its simplest forms it creates an
array of simple monomials:

.. code:: python

    >>> numpoly.monomial(5)
    polynomial([1, q0, q0**2, q0**3, q0**4])

It can be expanded to include number of dimensions and a lower bound for the
polynomial order:

.. code:: python

    >>> numpoly.monomial(start=2, stop=3, dimensions=2)
    polynomial([q0**2, q0*q1, q1**2])

Note that the polynomial is here truncated on total order, meaning that the sum
of exponents is limited by the :math:`L_1`-norm.
If a full tensor-product of polynomials, or another norm is wanted in the
truncation, this is also possible using the ``cross_truncation`` flag:

.. code:: python

    >>> numpoly.monomial(2, 3, dimensions=2, cross_truncation=numpy.inf)
    polynomial([q0**2, q0**2*q1, q1**2, q0*q1**2, q0**2*q1**2])
    >>> numpoly.monomial(2, 4, dimensions=2, cross_truncation=0.8)
    polynomial([q0**2, q0**3, q0*q1, q1**2, q1**3])
    >>> numpoly.monomial(2, 4, dimensions=2, cross_truncation=0.0)
    polynomial([q0**2, q0**3, q1**2, q1**3])

Alternative to the :func:`numpoly.monomial` function, it is also possible to
achieve the same expansion using the exponents only. For example:

.. code:: python

    >>> q0**numpy.arange(5)
    polynomial([1, q0, q0**2, q0**3, q0**4])

Or in the multivariate case:

.. code:: python

    >>> q0q1 = numpoly.variable(2)
    >>> expon = [[2, 0], [3, 0], [0, 2], [0, 3]]
    >>> numpoly.prod(q0q1**expon, axis=-1)
    polynomial([q0**2, q0**3, q1**2, q1**3])

To help construct these exponent, there is function :func:`numpoly.glexindex`.
It behave the same as :func:`numpoly.monomial`, but only creates the exponents.
E.g.:

.. code:: python

    >>> numpoly.glexindex(0, 5, 1).T
    array([[0, 1, 2, 3, 4]])
    >>> numpoly.glexindex(2, 3, 2, numpy.inf).T
    array([[2, 2, 0, 1, 2],
           [0, 1, 2, 2, 2]])
    >>> numpoly.glexindex(2, 4, 2, 0.8).T
    array([[2, 3, 1, 0, 0],
           [0, 0, 1, 2, 3]])
    >>> numpoly.glexindex(2, 4, 2, 0.0).T
    array([[2, 3, 0, 0],
           [0, 0, 2, 3]])

.. _numpy: https://numpy.org/doc/stable
