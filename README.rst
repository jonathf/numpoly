.. image:: doc/.static/numpoly_logo.svg
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

.. include:: doc/features.rst

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

In addition there are also several operators specific to the polynomial:

.. code-block:: python

   >>> numpoly.diff([1, x, x**2], x)
   polynomial([0, 1, 2*x])
   >>> numpoly.gradient([x*y, x+y])
   polynomial([[y, 1],
               [x, 1]])

Development
-----------

Development is done using `Poetry <https://poetry.eustace.io/>`_ manager.
Inside the repository directory, install and create a virtual environment with:

.. code-block:: bash

   poetry install

To run tests, run:

.. code-block:: bash

   poetry run pytest numpoly test doc --doctest-modules

Questions and Contributions
---------------------------

Please feel free to `file an issue
<https://github.com/jonathf/numpoly/issues>`_ for:

* report bugs.
* asking questions related to usage.
* requesting new feature.
* want to contribute with code.
