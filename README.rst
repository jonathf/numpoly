.. image:: doc/.static/numpoly_logo.svg
   :height: 300 px
   :width: 300 px
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

   >>> numpoly.monomial(start=0, stop=3, indeterminants=("x", "y"))
   polynomial([1, y, x, y**2, x*y, x**2, y**3, x*y**2, x**2*y, x**3])

It is also possible to construct your own from symbols:

.. code-block:: python

   >>> x, y = numpoly.symbols("x y")
   >>> numpoly.polynomial([1, x**2-1, x*y, y**2-1])
   polynomial([1, -1+x**2, x*y, -1+y**2])

Or in combination with other numpy objects:

.. code-block:: python

   >>> x**numpy.arange(4)-y**numpy.arange(3, -1, -1)
   polynomial([1-y**3, x-y**2, x**2-y, -1+x**3])

The polynomials can be evaluated as needed:

.. code-block:: python

   >>> poly = 3*x+2*y+1
   >>> poly(x=y, y=[1, 2, 3])
   polynomial([3+3*y, 5+3*y, 7+3*y])

The polynomials also support many numpy operations:

.. code-block:: python

   >>> numpy.reshape(x**numpy.arange(4), (2, 2))
   polynomial([[1, x],
               [x**2, x**3]])
   >>> numpy.sum(numpoly.monomial(12)[::3])
   polynomial(1+q**3+q**6+q**9+q**12)

There are also several polynomial specific operators:

.. code-block:: python

   >>> numpoly.diff([1, x, x**2], x)
   polynomial([0, 1, 2*x])
   >>> numpoly.gradient([x*y, x+y])
   polynomial([[y, 1],
               [x, 1]])


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
