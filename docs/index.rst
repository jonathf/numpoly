.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quick Overview:

   installation
   user_guide
   reference

numpoly -- numerical polynomials
================================

Numpoly is a generic library for creating, manipulating and evaluating
arrays of polynomials based on :class:`numpy.ndarray` objects.

The documentation consist of the following parts:

* `Installation instructions <./installation>`_ -- Instruction for getting
  ``numpoly`` installed on your system, both for user and developers.
* `User guide <./user_guide>`_ -- In-depth guides to the various concepts in
  ``numpoly``.
* `API reference <./reference>`_ -- The collection of public functions and
  classes.

At a glance
-----------

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

Questions, bug-reporting and contributing
-----------------------------------------

Please feel free to `file an issue
<https://github.com/jonathf/numpoly/issues>`_ for:

* bug reporting
* asking questions related to usage
* requesting new features
* wanting to contribute with code

.. _numpy: https://numpy.org/
