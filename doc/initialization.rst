Initialization
==============

The core element of the `numpoly` library is the `numpoly.ndpoly` class. The
class is subclass of of the `numpy.ndarray` and the implementation follows the
recommendation of how to subclass `numpy` objects.

In a nutshell, the `nmply.ndpoly` under the hood is a structured array, where
the column names represents the exponent powers as strings, and the values
represents the coefficients. In other words, the polynomial coefficients are
represented as `numpy.ndarray`:

.. math::

    \Phi(x_1, \dots, x_D) = \sum_{n=1}^N c_n x_1^{k_{1n}} \cdots x_D^{k_{Dn}}

Where :math:`\Phi` is polynomial vector, :math:`N` is the number of terms in the
polynomial sum, :math:`c_n` is a (potentially) multi-dimensional polynomial
coefficients, :math:`x_d` is the :math:`d`-th indeterminant name, and
:math:`k_{nd}` is the exponent for the :math:`n`-th polynomial term and the
:math:`d`-th indeterminant name.

For example, for a simple polynomial with scalar coefficients:

.. code:: python

    >>> x, y = numpoly.symbols("x y")
    >>> poly = numpoly.polynomial(4*x+3*y-1)
    >>> poly
    polynomial(3*y+4*x-1)
    >>> poly.coefficients
    [-1, 4, 3]
    >>> poly.exponents
    array([[0, 0],
           [1, 0],
           [0, 1]], dtype=uint32)
    >>> poly.indeterminants
    polynomial([x, y])

These three properties can be used to reconstruct the polynomial:

.. code:: python

    >>> terms = numpoly.prod(
    ...     poly.indeterminants**poly.exponents, -1)*poly.coefficients
    >>> terms
    polynomial([-1, 4*x, 3*y])
    >>> numpoly.sum(terms, 0)
    polynomial(3*y+4*x-1)

Underlying Structured Array
---------------------------

Under the hood, each of the polynomials are numpy structured arrays. The column
names are string representations corresponding to the polynomial exponents,
and the values are the coefficients. The indeterminant names are stored
separately. Numpoly is a wrapper on top of this.

From a development point of view, it is possible to view the underlying
structured array directly using the ``values`` attribute:

.. code:: python

    >>> array = poly.values
    >>> array  # doctest: +NORMALIZE_WHITESPACE
    array((-1, 4, 3),
          dtype=[(';;', '<i8'), ('<;', '<i8'), (';<', '<i8')])

Which, together with the indeterminant names, can be used to cast back the
array back to a polynomial:

.. code:: python

    >>> numpoly.aspolynomial(array, names=("x", "y"))
    polynomial(3*y+4*x-1)

Constructors
------------

.. autofunction:: numpoly.polynomial
.. autofunction:: numpoly.aspolynomial
.. autofunction:: numpoly.monomial
.. autofunction:: numpoly.symbols

.. autoclass:: numpoly.baseclass.ndpoly
    :members: coefficients, exponents, from_attributes, indeterminants, keys, names, values, __new__, __call__, _dtype, isconstant, todict, tonumpy
