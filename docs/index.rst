.. toctree::
   :hidden:

   user_guide/index.rst
   reference/index.rst
   about_us

Numpoly
=======

Numpoly is a generic library for creating, manipulating and evaluating
arrays of polynomials based on :class:`numpy.ndarray` objects.

The documentation consist of the following parts:


* Intuitive interface for users experienced with `numpy`_, as the library
  provides a high level of compatibility with the :class:`numpy.ndarray`,
  including fancy indexing, broadcasting, :class:`numpy.dtype`, vectorized
  operations to name a few.
* Computationally fast evaluations of lots of functionality inherent from
  `numpy`_.
* Vectorized polynomial evaluation.
* Support for arbitrary number of dimensions.
* Native support for lots of ``numpy.<name>`` functions using `numpy`_'s
  compatibility layer (which also exists as ``numpoly.<name>``
  equivalents).
* Support for polynomial division through the operators ``/``, ``%`` and
  ``divmod``.
* Extra polynomial specific attributes exposed on the polynomial objects like
  :attr:`~numpoly.ndpoly.exponents`, :attr:`~numpoly.ndpoly.coefficients`,
  :attr:`~numpoly.ndpoly.indeterminants` etc.
* Polynomial derivation through functions like :func:`numpoly.derivative`,
  :func:`numpoly.gradient`, :func:`numpoly.hessian` etc.
* Decompose polynomial sums into vector of addends usin
  :func:`numpoly.decompose`.
* Variable substitution through :func:`numpoly.call`.

.. _numpy: https://numpy.org/

.. _installation:

Installation
============

Installation should be straight forward from `pip <https://pypi.org/>`_:

.. code-block:: bash

    pip install numpoly

For developer installation, go to the `numpoly repository
<https://github.com/jonathf/numpoly>`_. Otherwise, check out the `user guide
<https://numpoly.readthedocs.io/en/master/user_guide>`_ to see how to use the
toolbox.
