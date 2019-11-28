.. _introduction:

Introduction
============

Numpoly is a generic library for creating, manipulating polynomial arrays.

Many numerical analysis, prominent in for example uncertainty quantification,
uses polynomial approximations as proxy for real models to do analysis on.
These models are often solutions to non-linear problems discretized with high
mesh. As such, the corresponding polynomial approximation consist of high
number of dimensions and large multi-dimensional polynomial coefficients.

The polynomial base class ``numpoly.ndpoly`` is a subclass of ``numpy.ndarray``
implemented to represent polynomials as array element. As such is fast and
scales very well with the size of the coefficients. It is also compatible with
most ``numpy`` functions, where that makes sense, making the interface fairly
intuitive. Some of the interface is also inspired by the ``sympy`` interface.

.. contents:: Table of Contents:

Installation
------------

Installation should be straight forward:

.. code-block:: bash

    pip install numpoly

And you should be ready to go.

Example usage
-------------

Constructing polynomial is typically done using one of the available
constructors:

.. code-block:: python

    >>> poly1 = numpoly.monomial(start=0, stop=4, names=("x", "y"))
    >>> poly1
    polynomial([1, y, x, y**2, x*y, x**2, y**3, x*y**2, x**2*y, x**3])

It is also possible to construct your own from symbols:

.. code-block:: python

    >>> x, y = numpoly.symbols("x y")
    >>> poly2 = numpoly.polynomial([1, x**2-1, x*y, y**2-1])
    >>> poly2
    polynomial([1, -1+x**2, x*y, -1+y**2])

Or in combination with other numpy objects:

.. code-block:: python

    >>> poly3 = x**numpy.arange(4)-y**numpy.arange(3, -1, -1)
    >>> poly3
    polynomial([1-y**3, x-y**2, x**2-y, -1+x**3])

The polynomials can be evaluated as needed:

.. code-block:: python

    >>> poly1(1, 2)
    array([1, 2, 1, 4, 2, 1, 8, 4, 2, 1])
    >>> poly2(x=[1, 2])
    polynomial([[1, 1],
                [0, 3],
                [y, 2*y],
                [-1+y**2, -1+y**2]])
    >>> poly1(x=0, y=2*x)
    polynomial([1, 2*x, 0, 4*x**2, 0, 0, 8*x**3, 0, 0, 0])

The polynomials also support many numpy operations:

.. code-block:: python

    >>> numpy.reshape(poly2, (2, 2))
    polynomial([[1, -1+x**2],
                [x*y, -1+y**2]])
    >>> poly1[::3].astype(float)
    polynomial([1.0, y**2, y**3, x**3])
    >>> numpy.sum(poly1.reshape(2, 5), 0)
    polynomial([1+x**2, y+y**3, x+x*y**2, y**2+x**2*y, x*y+x**3])

There are also several polynomial specific operators:

.. code-block:: python

    >>> numpoly.diff(poly3, y)
    polynomial([-3*y**2, -2*y, -1, 0])
    >>> numpoly.gradient(poly3)
    polynomial([[0, 1, 2*x, 3*x**2],
                [-3*y**2, -2*y, -1, 0]])

Rational
--------

The main reason for creating this is because I need it as a backend component
for the `chaospy <https://github.com/jonathf/chaospy>`_ library. It can be
replaced by alternative software, but for its particular requirements, building
something from scratch made the most sense.

* Why not `numpy.polynomial <https://docs.scipy.org/doc/numpy/reference/routines.polynomials.polynomial.html>`_?

  The numpy native polynomial class is likely better at what it does, but it is
  limited to only 3 dimensions. This makes it a non-starter as a backend for
  ``chaospy``.

* Why not `sympy <https://www.sympy.org>`_?

  ``sympy`` is a great option that can do the same as ``numpoly`` and quite
  a bit more. However it is not using the vectorization utilized by ``numpy``
  and relies on pure python for its operations. A process notably slower than
  what it could be in many instances.

Development
-----------

Development is done using `Poetry <https://poetry.eustace.io/>`_ manager.
Inside the repository directory, install and create a virtual environment with:

.. code-block:: bash

   poetry install

To run tests, run:

.. code-block:: bash

   poentry run pytest numpoly test doc --doctest-modules

Questions & Troubleshooting
---------------------------

For any problems and questions you might have related to ``numpoly``, please
feel free to file an `issue <https://github.com/jonathf/numpoly/issues>`_.
