.. _introduction:

Introduction
============

Numpoly is a generic library for creating, manipulating and evaluating
arrays of polynomials.

The polynomial base class ``numpoly.ndpoly`` is a subclass of ``numpy.ndarray``
implemented to represent polynomials as array element. This makes the library
very fast with the respect of the size of the coefficients. It is also adds
compatibility with ``numpy`` functions and methods, where that makes sense,
making the interface more intuitive.

Many numerical analysis, polynomial approximations as proxy predictors for real
predictors to do analysis on. These models are often solutions to non-linear
problems discretized with high mesh. As such, the corresponding polynomial
approximation consist of high number of dimensions and large multi-dimensional
polynomial coefficients. For these kind of problems ``numpoly`` is a good fit.

Feature Overview
----------------

* Intuitive interface for users experienced with ``numpy``, as the library
  provides a high level of compatibility with the ``numpy.ndarray``, including
  fancy indexing, broadcasting, ``numpy.dtype``, vectorized operations to name
  a few.
* Computationally fast evaluations of lots of functionality inherent from
  ``numpy``.
* Vectorized polynomial evaluation.
* Support for arbitrary number of dimensions and name for the indeterminants.
* Native support for lots of ``numpy.<name>`` functions using ``numpy``'s
  compatibility layer (which also exists as ``numpoly.<name>``
  equivalents).
* Support for polynomial division through the operators ``/``, ``%`` and
  ``divmod``.
* Extra polynomial specific attributes exposed on the polynomial objects like
  ``poly.exponents``, ``poly.coefficients``, ``poly.indeterminants`` etc.
* Polynomial derivation through functions like ``numpoly.diff``,
  ``numpoly.gradient``, ``numpoly.hessian`` etc.
* Decompose polynomial sums into vector of addends using ``numpoly.decompose``.
* Variable substitution through ``numpoly.call``.

Installation
------------

Installation should be straight forward:

.. code-block:: bash

    pip install numpoly

And you should be ready to go. That is it. You should now be able to import the
library in your Python REPL:

.. code-block:: python

    >>> import numpoly

Example Usage
-------------

Constructing polynomial is typically done using one of the available
constructors:

.. code-block:: python

   >>> numpoly.monomial(start=0, stop=4, names=("x", "y"))
   polynomial([1, y, x, y**2, x*y, x**2, y**3, x*y**2, x**2*y, x**3])

It is also possible to construct your own from symbols:

.. code-block:: python

   >>> x, y = numpoly.symbols("x y")
   >>> numpoly.polynomial([1, x**2-1, x*y, y**2-1])
   polynomial([1, -1+x**2, x*y, -1+y**2])

Or in combination with numpy objects using various arithmetics:

.. code-block:: python

   >>> x**numpy.arange(4)-y**numpy.arange(3, -1, -1)
   polynomial([1-y**3, x-y**2, x**2-y, -1+x**3])

The constructed polynomials can be evaluated as needed:

.. code-block:: python

   >>> poly = 3*x+2*y+1
   >>> poly(x=y, y=[1, 2, 3])
   polynomial([3+3*y, 5+3*y, 7+3*y])

Or manipulated using various numpy functions:

.. code-block:: python

   >>> numpy.reshape(x**numpy.arange(4), (2, 2))
   polynomial([[1, x],
               [x**2, x**3]])
   >>> numpy.sum(numpoly.monomial(13, names="z")[::3])
   polynomial(1+z**3+z**6+z**9+z**12)
