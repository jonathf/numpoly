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

``numpoly`` is currently being used as the backend is the uncertainty
quantification library `chaospy <https://github.com/jonathf/chaospy>`_.

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
    polynomial([1, x, x**2, x**3, y, x*y, x**2*y, y**2, x*y**2, y**3])

It is also possible to construct your own from symbols:

.. code-block:: python

    >>> x, y = numpoly.symbols("x y")
    >>> numpoly.polynomial([1, x**2-1, x*y, y**2-1])
    polynomial([1, x**2-1, x*y, y**2-1])

Or in combination with numpy objects using various arithmetics:

.. code-block:: python

    >>> x**numpy.arange(4)-y**numpy.arange(3, -1, -1)
    polynomial([-y**3+1, -y**2+x, x**2-y, x**3-1])

The constructed polynomials can be evaluated as needed:

.. code-block:: python

    >>> poly = 3*x+2*y+1
    >>> poly(x=y, y=[1, 2, 3])
    polynomial([3*y+3, 3*y+5, 3*y+7])

Or manipulated using various numpy functions:

.. code-block:: python

    >>> numpy.reshape(x**numpy.arange(4), (2, 2))
    polynomial([[1, x],
                [x**2, x**3]])
    >>> numpy.sum(numpoly.monomial(13, names="z")[::3])
    polynomial(z**12+z**9+z**6+z**3+1)

Development
-----------

Development is done using `Poetry <https://poetry.eustace.io/>`_ manager.
Inside the repository directory, install and create a virtual environment with:

.. code-block:: bash

   poetry install

To run tests:

.. code-block:: bash

   poetry run pytest numpoly test doc --doctest-modules

To build documentation, run:

.. code-block:: bash

   cd doc/
   make html

The documentation will be generated into the folder ``doc/.build/html``.

Questions and Contributions
---------------------------

Please feel free to `file an issue
<https://github.com/jonathf/numpoly/issues>`_ for:

* bug reporting
* asking questions related to usage
* requesting new features
* wanting to contribute with code
