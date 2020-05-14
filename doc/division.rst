Division
========

Numerical division can be split into two variants: floor division and true
division:

.. code:: python

    >>> dividend = 7
    >>> divisor = 2
    >>> quotient_true = numpy.true_divide(dividend, divisor)
    >>> quotient_true
    3.5
    >>> quotient_floor = numpy.floor_divide(dividend, divisor)
    >>> quotient_floor
    3

The discrepancy between the two can be captured by a remainder, which allow us
to more formally define them as follows:

.. code:: python

    >>> remainder = numpy.remainder(dividend, divisor)
    >>> remainder
    1
    >>> dividend == quotient_floor*divisor+remainder
    True
    >>> dividend == quotient_true*divisor
    True


In the case of polynomials, neither true nor floor division is supported like
this. Instead it support its own kind of polynomial division. Polynomial
division falls back to behave like floor division for all constants, as it does
not round values:

.. code:: python

    >>> x, y = numpoly.symbols("x y")
    >>> dividend = x**2+y
    >>> divisor = x-1
    >>> quotient = numpoly.poly_divide(dividend, divisor)
    >>> quotient
    polynomial(1.0+x)

However, like floor division, it can still have remainders.
For example:

.. code:: python

    >>> remainder = numpoly.poly_remainder(dividend, divisor)
    >>> remainder
    polynomial(y+1.0)
    >>> dividend == quotient*divisor+remainder
    True

In ``numpy``, the "syntactic sugar" operators:

* ``/`` is used for true division.
* ``//`` is used for floor division.
* ``%`` is used for remainder.
* ``divmod`` is used for floor division and remainder in combination to save
  computational cost.

In ``numpoly``, which takes precedence if any of the values are of
``numpoly.ndpoly`` objects:

* ``/`` is used for polynomial division, which is backwards compatible with
  ``numpy``.
* ``//`` is still used for floor division as in ``numpy``, which is only
  possible if divisor is a constant.
* ``%`` is used for polynomial remainder, which is not backwards compatible.
* ``divmod`` is used for polynomial division and remainder in combination to
  save computation cost.

.. autofunction:: numpoly.poly_function.poly_divide
.. autofunction:: numpoly.poly_function.poly_divmod
.. autofunction:: numpoly.poly_function.poly_remainder
