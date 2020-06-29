Introduction
============

In `numpy`_ the concept of *array* is generalized to imply arrays of arbitrary
dimension, overlapping with the concept of *scalars*, *matrices* and *tensors*.
To allow arrays of various dimensions to operate together it defines
unambiguous broadcasting rules for what to expect. The results is a library
that is used as the reference for almost all of the numerical Python community.

In mathematical literature the term *polynomial expansions* is used to denote a
collection of polynomials. Though they strictly do not need to, they are often
indexed, giving each polynomial both a label and a position for where to locate
a polynomial relative to the others. Assuming that there always is an index,
one could say that *polynomial expansions* could just as well be termed
*polynomial array*. And using the rules defined in `numpy`_, there no reason
not to also start talking about *multi-dimensional polynomial arrays*.

The main idea here is that in the same way as :class:`numpy.ndarray` are
composed of scalars, :class:`ndpoly <numpoly.baseclass.ndpoly>` --
the baseclass for the polynomial arrays -- are composed of simpler polynomials.
For example:

.. code:: python

    >>> q0, q1 = numpoly.variable(2)
    >>> poly = q1**2-q0
    >>> poly
    polynomial(q1**2-q0)
    >>> expansion = numpoly.polynomial([1, q0, q1**2])
    >>> expansion
    polynomial([1, q0, q1**2])

This gives a convenient interface for dealing with many polynomials at the same
time.

The ``numpoly`` concept of arrays, which is taken from `numpy`_ goes deeper
than just inspiration. The base class
:class:`numpoly.ndpoly <numpoly.baseclass.ndpoly>` is a direct subclass of
:class:`numpy.ndarray`:

.. code:: python

    >>> issubclass(numpoly.ndpoly, numpy.ndarray)
    True

The intentions is to have a library that is fast with the respect of the number
of coefficients, as it leverages `numpy`_'s speed where possible.

In addition ``numpoly`` is designed to be behave both as you would expect as a
polynomial, but also, where possible, to behave as a `numpy`_ numerical array.
In practice this means that ``numpoly`` provides a lot functions that also
exists in `numpy`_, which does about the same thing. If one of these ``numpoly``
function is provided with a :class:`numpy.ndarray` object, the returned values
is the same as if provided to the `numpy`_ function with the same name. For
example:

.. code:: python

    >>> num_array = numpy.array([[1, 2], [3, 4]])
    >>> numpoly.transpose(num_array)
    polynomial([[1, 3],
                [2, 4]])

And this works the other way around as well. If a polynomial is provided to the
`numpy`_ function, it will behave the same way as if it was provided to the
``numpoly`` equivalent. For example:

.. code:: python

    >>> poly_array = numpoly.polynomial([[1, q0-1], [q1**2, 4]])
    >>> numpy.transpose(poly_array)
    polynomial([[1, q1**2],
                [q0-1, 4]])

Though the compatibility layer between `numpy`_ and ``numpoly`` is large, there
are still lots of functionality which is polynomial specific. The most obvious,
is the ability to evaluate the polynomials:

.. code:: python

    >>> poly
    polynomial(q1**2-q0)
    >>> poly(4, 4)
    12
    >>> poly(4)
    polynomial(q1**2-4)
    >>> poly([1, 2, 3])
    polynomial([q1**2-1, q1**2-2, q1**2-3])

.. _numpy: https://numpy.org/doc/stable
