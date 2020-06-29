Initialization
==============

A multivariate polynomial can more formally be defined as follows:

.. math::

    \Phi(q_1, \dots, q_D) = \sum_{n=1}^N c_n q_1^{k_{1n}} \cdots q_D^{k_{Dn}}

Where :math:`\Phi` is polynomial vector, :math:`N` is the number of terms in
the polynomial sum, :math:`c_n` is a (potentially) multi-dimensional polynomial
coefficients, :math:`q_d` is the :math:`d`-th indeterminant name, and
:math:`k_{nd}` is the exponent for the :math:`n`-th polynomial term and the
:math:`d`-th indeterminant name.

This means that to uniquely defined a polynomial, we need the three components:
``coefficients``, ``exponents`` and ``indeterminants``.
both conceptually and visually in the polynomial string representation, we
might conclude that these three attributes are define locally per cell.
However, to be able to utilize the speed of `numpy`_ vectorization, these three
properties are global with respect to the array.

We can access these three defining properties directly from
:class:`~numpoly.baseclass.ndpoly`.
For example, for a simple polynomial with scalar coefficients:

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

Because these three properties uniquely define a polynomial array, they can be
used to reconstruct the polynomial:

.. code:: python

    >>> terms = coeff*numpoly.prod(indet**expon, -1)
    >>> terms
    polynomial([-1, 4*q0, 3*q1])
    >>> poly = numpoly.sum(terms, axis=0)
    >>> poly
    polynomial(3*q1+4*q0-1)

.. _numpy: https://numpy.org/doc/stable

Constructors
------------

The examples above uses the idea of simple variables from
:func:`~numpoly.variable` and initialization function
:func:`~numpoly.polynomial` as the foundation for
constructing a polynomial arrays. However, there are other similar functions
that can be used:

.. autofunction:: numpoly.variable
.. autofunction:: numpoly.polynomial
.. autofunction:: numpoly.aspolynomial
.. autofunction:: numpoly.monomial
.. autofunction:: numpoly.symbols
.. autofunction:: numpoly.polynomial_from_attributes

Baseclass
---------

.. autoclass:: numpoly.baseclass.ndpoly
    :members: coefficients, exponents, from_attributes, indeterminants, keys, names, values, __new__, __call__, _dtype, isconstant, todict, tonumpy
