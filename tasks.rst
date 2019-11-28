Tasks that needs solving
========================

Some undone task that could be be useful to add. Any help is welcome.

Please create a Github issue if you want to claim a task.

Adding interfaces for numpy functionality
-----------------------------------------

Difficulty: easy-medium-hard

Lots of numpy functions can use wrappers to get a more whole numpy-like
experience. The task should include both a wrapper and a corresponding test,
testing the feature.

Sub-tasks can individually be claimed.

===================  ==========================================================
Function name        Claimed by
===================  ==========================================================
amax
amin
average
broadcast*
cast
clip
count_nonzero
cov
cross
cumprod
diag
mean
median
nonzero
ravel
repeat
resize
split
===================  ==========================================================

Save/Load support
-----------------

Difficulty: easy

Pickle should work out of the box. Preferably using the ``np.save``
interface. But if not possible, using something like h5py.

Along the way:
* Add optional requirement h5py (if applicable)
* Add test demonstrating the feature

True division and mod support
-----------------------------

Difficulty: medium-hard

Currently division is only supported if the divisor is ndarray. However,
polynomial division is a thing. And limiting the division to true-division
``//`` and division mod ``%``, allows for the result to ensured to be
a polynomial as well.

Some exploration is needed to find out how feasible multi-variate support is.
Not creating a for-loop over the numpy elements is preferable.

``power`` implementation
------------------------

Difficulty: medium-hard

Current implementation is a hack: repeat multiplying against itself ``n``
times. For-loop over ``n``, if array.

* Create a method that does not rely on repeated calling multiply, and instead
  allocates a single chunk of memory and fill inn results there.
* Somehow avoid using element-by-element for-loop over exponents.

``prod`` and ``cumprod`` implementation
---------------------------------------

Difficulty: hard

Current implementation uses multiply repeatedly along axis. This can likely be
done much more efficiently with dedicated code.

Somewhat the same problem as ``power``, but a lot more book keeping.


Element-in support (``x in y``)
-------------------------------

Difficulty: medium

This is a bit undefined. What does it mean to have a polynomial as an element
in another polynomial? Given a reasonable definition, implement such that it
works.

Element-in support (``isin(x, y)``)
-----------------------------------

Difficulty: medium

Maybe same as above, depends on the definition used.

Inner product support
---------------------

Difficulty: medium-hard

The definition of inner production of tensor products.
