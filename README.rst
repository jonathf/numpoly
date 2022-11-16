.. image:: https://github.com/jonathf/numpoly/raw/master/docs/.static/numpoly_logo.svg
   :height: 200 px
   :width: 200 px
   :align: center

|circleci| |codecov| |readthedocs| |downloads| |pypi|

.. |circleci| image:: https://circleci.com/gh/jonathf/numpoly/tree/master.svg?style=shield
    :target: https://circleci.com/gh/jonathf/numpoly/tree/master
.. |codecov| image:: https://codecov.io/gh/jonathf/numpoly/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jonathf/numpoly
.. |readthedocs| image:: https://readthedocs.org/projects/numpoly/badge/?version=master
    :target: http://numpoly.readthedocs.io/en/master/?badge=master
.. |downloads| image:: https://img.shields.io/pypi/dm/numpoly
    :target: https://pypistats.org/packages/numpoly
.. |pypi| image:: https://badge.fury.io/py/numpoly.svg
    :target: https://badge.fury.io/py/numpoly

Numpoly is a generic library for creating, manipulating and evaluating
arrays of polynomials based on ``numpy.ndarray`` objects.

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
============

Installation should be straight forward:

.. code-block:: bash

    pip install numpoly

Example Usage
=============

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

Installation
============

Installation should be straight forward from `pip <https://pypi.org/>`_:

.. code-block:: bash

    pip install numpoly

Alternatively, to get the most current experimental version, the code can be
installed from `Github <https://github.com/>`_ as follows:

* First time around, download the repository:

  .. code-block:: bash

      git clone git@github.com:jonathf/numpoly.git

* Every time, move into the repository:

  .. code-block:: bash

      cd numpoly/

* After  the first time, you want to update the branch to the most current
  version of ``master``:

  .. code-block:: bash

      git checkout master
      git pull

* Install the latest version of ``numpoly`` with:

  .. code-block:: bash

      pip install .

Development
-----------

Installing ``numpoly`` for development can
be done from the repository root with the command::

    pip install -e .[dev]

The deployment of the code is done with Python 3.10 and dependencies are then
fixed using::

    pip install -r requirements-dev.txt

Testing
-------

To run test:

.. code-block:: bash

    pytest --doctest-modules numpoly test docs/user_guide/*.rst README.rst

Documentation
-------------

To build documentation locally on your system, use ``make`` from the ``doc/``
folder:

.. code-block:: bash

    cd doc/
    make html

Run ``make`` without argument to get a list of build targets. All targets
stores output to the folder ``doc/.build/html``.

Note that the documentation build assumes that ``pandoc`` is installed on your
system and available in your path.
