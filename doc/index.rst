NumPoly's Documentation
=======================

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   initialization
   division
   comparison
   derivative
   array_function
