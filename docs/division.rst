Polynomial Division
===================

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

    >>> q0, q1 = numpoly.variable(2)
    >>> dividend = q0**2+q1
    >>> divisor = q0-1
    >>> quotient = numpoly.poly_divide(dividend, divisor)
    >>> quotient
    polynomial(q0+1.0)

However, like floor division, it can still have remainders.
For example:

.. code:: python

    >>> remainder = numpoly.poly_remainder(dividend, divisor)
    >>> remainder
    polynomial(q1+1.0)
    >>> dividend == quotient*divisor+remainder
    True

In ``numpy``, the "Python syntactic sugar" operators have the following
behavior:

* ``/`` is used for true division.
* ``//`` is used for floor division.
* ``%`` is used for remainder.
* ``divmod`` is used for floor division and remainder in combination to save
  computational cost.

In ``numpoly``, which takes precedence if any of the values are of
``numpoly.ndpoly`` objects, take the following behavior:

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
