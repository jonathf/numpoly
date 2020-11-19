.. image:: https://github.com/jonathf/numpoly/raw/master/docs/.static/numpoly_logo.svg
   :height: 200 px
   :width: 200 px
   :align: center

|circleci| |codecov| |pypi| |readthedocs|

.. |circleci| image:: https://circleci.com/gh/jonathf/numpoly/tree/master.svg?style=shield
    :target: https://circleci.com/gh/jonathf/numpoly/tree/master
.. |codecov| image:: https://codecov.io/gh/jonathf/numpoly/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jonathf/numpoly
.. |pypi| image:: https://badge.fury.io/py/numpoly.svg
    :target: https://badge.fury.io/py/numpoly
.. |readthedocs| image:: https://readthedocs.org/projects/numpoly/badge/?version=master
    :target: http://numpoly.readthedocs.io/en/master/?badge=master

Numpoly is a generic library for creating, manipulating and evaluating
arrays of polynomials based on ``numpy.ndarray`` objects.

.. contents:: Table of Contents:

Feature Overview
----------------

* Intuitive interface for users experienced with ``numpy``, as the library
  provides a high level of compatibility with the ``numpy.ndarray``, including
  fancy indexing, broadcasting, ``numpy.dtype``, vectorized operations to name
  a few.
* Computationally fast evaluations of lots of functionality inherent from
  ``numpy``.
* Vectorized polynomial evaluation.
* Support for arbitrary number of dimensions.
* Native support for lots of ``numpy.<name>`` functions using ``numpy``'s
  compatibility layer (which also exists as ``numpoly.<name>``
  equivalents).
* Support for polynomial division through the operators ``/``, ``%`` and
  ``divmod``.
* Extra polynomial specific attributes exposed on the polynomial objects like
  ``poly.exponents``, ``poly.coefficients``, ``poly.indeterminants`` etc.
* Polynomial derivation through functions like ``numpoly.derivative``,
  ``numpoly.gradient``, ``numpoly.hessian`` etc.
* Decompose polynomial sums into vector of addends using ``numpoly.decompose``.
* Variable substitution through ``numpoly.call``.

Installation
------------

Installation should be straight forward:

.. code-block:: bash

    pip install numpoly

Example Usage
-------------

Constructing polynomial is typically done using one of the available
constructors:

.. code-block:: python

    >>> import numpoly
    >>> numpoly.monomial(start=0, stop=3, dimensions=2)
    polynomial([1, q0, q0**2, q1, q0*q1, q1**2])

It is also possible to construct your own from symbols together with
`numpy <https://python.org>`_:

.. code-block:: python

    >>> import numpy
    >>> q0, q1 = numpoly.variable(2)
    >>> numpoly.polynomial([1, q0**2-1, q0*q1, q1**2-1])
    polynomial([1, q0**2-1, q0*q1, q1**2-1])

Or in combination with numpy objects using various arithmetics:

.. code-block:: python

    >>> q0**numpy.arange(4)-q1**numpy.arange(3, -1, -1)
    polynomial([-q1**3+1, -q1**2+q0, q0**2-q1, q0**3-1])

The constructed polynomials can be evaluated as needed:

.. code-block:: python

    >>> poly = 3*q0+2*q1+1
    >>> poly(q0=q1, q1=[1, 2, 3])
    polynomial([3*q1+3, 3*q1+5, 3*q1+7])

Or manipulated using various numpy functions:

.. code-block:: python

    >>> numpy.reshape(q0**numpy.arange(4), (2, 2))
    polynomial([[1, q0],
                [q0**2, q0**3]])
    >>> numpy.sum(numpoly.monomial(13)[::3])
    polynomial(q0**12+q0**9+q0**6+q0**3+1)
