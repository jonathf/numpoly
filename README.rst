Numpoly is a generic library for creating, manipulating polynomial arrays.

Many numerical analysis, prominent in for example uncertainty quantification,
uses polynomial approximations as proxy for real models to do analysis on.
These models are often solutions to non-linear problems discretized with high
mesh. As such, the corresponding polynomial approximation consist of high
number of dimensions and large multi-dimensional polynomial coefficients.

``numpoly`` is a subclass of ``numpy.ndarray`` implemented to represent
polynomials as array element. As such is fast and scales very well with the
size of the coefficients. It is also compatible with most ``numpy`` functions,
where that makes sense, making the interface fairly intuitive. Some of the
interface is also inspired by the ``sympy`` interface.

|circleci| |codecov| |pypi|

.. |circleci| image:: https://circleci.com/gh/jonathf/numpoly/tree/master.svg?style=shield
    :target: https://circleci.com/gh/jonathf/numpoly/tree/master
.. |codecov| image:: https://codecov.io/gh/jonathf/numpoly/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jonathf/numpoly
.. |pypi| image:: https://badge.fury.io/py/numpoly.svg
    :target: https://badge.fury.io/py/numpoly

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

   >>> poly1 = numpoly.monomial(("x", "y"), start=0, stop=3)
   >>> print(poly1)
   [1 y x x*y x**2 y**2 y**3 x*y**2 x**2*y x**3]

It is also possible to construct your own from symbols:

.. code-block:: python

   >>> x, y = numpoly.symbols("x y")
   >>> poly2 = numpoly.polynomial([1, x**2-1, x*y, y**2-1])
   >>> print(poly2)
   [1 -1+x**2 x*y -1+y**2]

Or in combination with other numpy objects:

.. code-block:: python

   >>> poly3 = x**numpy.arange(4)-y**numpy.arange(3, -1, -1)
   >>> print(poly3)
   [1-y**3 -y**2+x -y+x**2 -1+x**3]

The polynomials can be evaluated as needed:

.. code-block:: python

   >>> print(poly1(1, 2))
   [1 2 1 2 1 4 8 4 2 1]
   >>> print(poly2(x=[1, 2]))
   [[1 1]
    [0 3]
    [y 2*y]
    [-1+y**2 -1+y**2]]
   >>> print(poly1(x=y, y=2*x))
   [1 2*x y 2*x*y y**2 4*x**2 8*x**3 4*x**2*y 2*x*y**2 y**3]

The polynomials also support many numpy operations:

.. code-block:: python

   >>> print(numpy.reshape(poly2, (2, 2)))
   [[1 -1+x**2]
    [x*y -1+y**2]]
   >>> print(poly1[::3].astype(float))
   [1.0 x*y y**3 x**3]
   >>> print(numpy.sum(poly1.reshape(2, 5), 0))
   [1+y**2 y+y**3 x+x*y**2 x*y+x**2*y x**2+x**3]

There are also several polynomial specific operators:

.. code-block:: python

   >>> print(numpoly.diff(poly3, y))
   [-3*y**2 -2*y -1 0]
   >>> print(numpoly.gradient(poly3))
   [[0 1 2*x 3*x**2]
    [-3*y**2 -2*y -1 0]]


Development
-----------

Development is done using `Poetry <https://poetry.eustace.io/>`_ manager.
Inside the repository directory, install and create a virtual enviroment with:

.. code-block:: bash

   poetry install

To run tests, run:

.. code-block:: bash

   poentry run pytest numpoly test --doctest-modules

Questions & Troubleshooting
---------------------------

For any problems and questions you might have related to ``numpoly``, please
feel free to file an `issue <https://github.com/jonathf/numpoly/issues>`_.
