Tasks that needs solving
========================

Some undone task that could be be useful to add. Any help is welcome.

Please create a Github issue if you want to claim a task.

Save/Load support
-----------------

Difficulty: easy

Pickle should work out of the box. Preferably using the ``np.save``
interface. But if not possible, using something like h5py.

Along the way:
* Add optional requirement h5py (if applicable)
* Add test demonstrating the feature

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

The definition of inner product of tensor products.
