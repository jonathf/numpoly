Master Branch
=============

Version 1.0.6 (2020-07-27)
==========================

* First iteration for deprecating `align_shape`:
  * Added common `assert_equal` function to test contiguous-ness, shape,
    dtype etc. in a more automated way.
  * Small patches here and there to assert contiguous-ness.
* Tests for `numpoly.equal`.
* Rename single variable differentiation: `diff -> derivative` (as the former
  is reserved in numpy).
* Numpy function additions: `result_type`, `diff`.

Version 1.0.5 (2020-07-08)
==========================

* Bugfix: Poly-division with large relative error caused infinity-loops.
* Pickle support.
* Rename function `largest_exponent -> lead_exponent`.
* Polynomial function addition: `lead_coefficient`.
* (Unwrapped) numpy function addition: `load`, `loadtxt`.
* Numpy function additions: `save`, `savetxt`, `savez`, `savez_compressed`.

Version 1.0.4 (2020-07-01)
==========================

* Numpy function additions: `diag`, `diagonal`, `ones`
* Added changelog (the file you currently are reading).

Version 1.0.3 (2020-06-26)
==========================

* Bugfix in `set_dimensions` (1.0.2 solution didn't work).

Version 1.0.2 (2020-06-26)
==========================

* Bugfix in `set_dimensions`.

Version 1.0.1 (2020-06-26)
==========================

* Polynomial function addition: `variable`, `set_dimensions`.

Version 1.0.0 (2020-06-26)
==========================

* Enforce all polynomials on the format `"q\d+"` aligning the defaults with
  chaospy.
* Lots and lots of doctests updated.

Version 0.3.0 (2020-06-08)
==========================

Comparison operator support!

* String representation changed from insertion order to Graded-reverse
  lexicographically.
* Move global options into common file with interfaces:
  `get_options`, `set_options` and `global_options` (context manager).
* New comparison and poly-division and derivative chapter in Sphinx, instead of
  poly-functions.
* Polynomial function additions:
  `poly_divide`, `poly_divmod`, `poly_reminder`, `sortable_proxy`.
* Numpy function additions:
  `amax`, `amin`, `argmin`. `argmax`, `greater`, `greater_equal`,
  `less`, `less_equal`, `max`, `maximum`, `min`, `minimum`.
* Lots of extra tests.

Version 0.2.3 (2020-05-08)
==========================

* More aggressive cross-truncation approach to keep memory low under `bindex`.

Version 0.2.2 (2020-05-08)
==========================

* More documentation updates.
* Cleanup to `bindex` for better handle of implicit dimensions.

Version 0.2.1 (2020-05-06)
==========================

* Documentation updates.

Version 0.2.0 (2020-05-05)
==========================

Polynomial division support!

* Polynomial function additions: `poly_divide`, `poly_divmod`, `poly_reminder`.
* Numpy function additions:
  `true_divide`, `divmod`, `remainder`. `where`, `zeros`.

Version 0.1.17 (2020-04-29)
===========================

* Numpy function additions:
  `apply_along_axis`, `apply_over_axes`, `expand_dims`.

Version 0.1.16 (2020-04-18)
===========================

* Bugfix to `bindex` in handle of indices around 0.

Version 0.1.15 (2020-04-17)
===========================

* Support for enforced naming convention (for use in Chaospy).

Version 0.1.14 (2020-04-15)
===========================

* Refactor of `monomial` again to deal with speed issue.

Version 0.1.13 (2020-03-31)
===========================

* Remove CircleCI cache for py2 (as it is really light).
* Better py2 support.
* Allow for debugging messages through `$NUMPOLY_DEBUG` environmental variable.
* Numpy function additions: `count_nonzero`, `nonzero` (thanks Fredrik Meyer)
* Add version number to `numpoly.__version__`.

Version 0.1.12 (2020-03-02)
===========================

* Bugfix in `prod`.

Version 0.1.11 (2020-02-26)
===========================

* Support for the "empty set" polynomial: `polynomial([])`.

Version 0.1.10 (2020-02-26)
===========================

* Refactor `monomial`, cleaning it out and catching some subtle bugs.

Version 0.1.9 (2020-02-26)
==========================

* Documentation cleanup.
* Small bugfix in `monomial` in how it implicitly handles multiple dimensions.

Version 0.1.8 (2020-02-24)
==========================

* Numpy function additions: `matmul`.

Version 0.1.7 (2020-02-11)
==========================

* Numpy function additions: `broadcast_arrays`.

Version 0.1.6 (2020-01-10)
==========================

* Small bugfix in experimental code.

Version 0.1.5 (2020-01-10)
==========================

* Move key index offset from 48 (the visually appealing 0, 1, 2, ...)
  to 59 (skipping the problematic 58 ':').
* Documentation update.
* Numpy function additions:
  `array_split`, `dsplit`, `hsplit`, `split`, `vsplit`.

Version 0.1.4 (2019-12-01)
==========================

* Numpy function additions: `tile` (failed to be added in 0.1.3).

Version 0.1.3 (2019-12-01)
==========================

* More documentation.
* Rename function arg `{indeterminants -> names}` (all over the place).
* Numpy function additions: `transpose`, `tile`.

Version 0.1.2 (2019-11-26)
==========================

* Doctest root readme on CircleCI.
* New Numpoly logo.
* Introduction chapter added to Sphinx.
* Numpy function additions: `choose`, `reshape`.
* Collection global constant into single dictionary.

Version 0.1.1 (2019-11-21)
==========================

* Small documentation updates.

Version 0.1.0 (2019-11-17)
==========================

* Support for alpha, beta, rc, dev, post releases.
* Validate tags against install version.
* Polynomial function addition: `decompose`.

Version 0.0.17 (2019-10-20)
===========================

* Change string representation to display polynomial
  by insertion order (affecting a lot of examples).

Version 0.0.16 (2019-10-01)
===========================

* Global constants added for manipulating string representation.
* Change `monomial` to have `indeterminants` argument at the end.

Version 0.0.15 (2019-09-27)
===========================

* Include Sphinx docs in CircleCI testing.
* Add Construct chapter to Sphinx.
* Rename `toarray -> tonumpy`, `as_ndarray -> values` (function -> property).
* Numpy function additions: `repeat`.

Version 0.0.14 (2019-09-27)
===========================

* Documentation update:
  * Introduction to `ndpoly` baseclass added.
  * Polynomial function collection.
  * Enforce complete function list through `sphinx_automodapi`.
  * Read-the-docs deployment configuration.
* Remove functions mappings between exponents and keys in favor of in-line
  solution.
* Bugfixes and code cleanups for `concatenate`, `*stack` and multiplications.
* `ndpoly` method additions: `as_ndarray`.
* Numpy function additions: `stack`.

Version 0.0.13 (2019-09-25)
===========================

* CircleCI cleanup: limited py27 and full py37 testing only.
* First iteration Sphinx docs.
* Increased testing coverage.
* Numpy function additions: `atleast_1d`, `atleast_2d`, `atleast_3d`,
  `ceil`, `floor`, `dstack`, `hstack`, `vstack`.

Version 0.0.12 (2019-09-13)
===========================

* Add align_dtype to alignment process.
* Recast dtype support in `ndpoly.__call__` when input is other format than
  internal one.

Version 0.0.11 (2019-09-12)
===========================

* Move testing dispatching to `conftest.py`.
* Numpy function additions: `prod`, `moveaxis`.
* Testing polish.

Version 0.0.10 (2019-09-12)
===========================

* Variable name typo fixes.
* Testing of alignment.
* Split testing suite into py2 and py3
  (as py3 supports full dispatching, and py2 does not).
* Numpy function additions: `allclose`, `isclose`, `isfinite`, `mean`.

Version 0.0.9 (2019-09-12)
==========================

* Linting added to CircleCI checks.
* Some code clean-up of alignment.
* Added `simple_dispatch` function to unify the backend for the most simplest
  numpy functions.
* Refactor constructions functions.
* Renamings: `ndpoly.{_exponents -> keys}`, `ndpoly.{_indeterminants -> names}`,
  `numpy.{clean_polynomial_attributes -> clean_attributes}`
* Support for numpy reduce and accumulate mappings.
* `ndpoly` method additions: `from_attributes`,
  `round` (likely needed because of numpy bug).
* Numpy function addition: `logical_and`, `rind`, `square`.

Version 0.0.8 (2019-09-11)
==========================

* Functions for mapping between `Tuple[int, ...]` and `str` for
  dealing with exponents, instead of using exposed maps.
* Split array functions into one-file-per-function.
* Polynomial function addition: `aspolynomial`.
* Numpy function addition: `around`, `common_type`, `inner`, `logical_or`.

Version 0.0.7 (2019-09-08)
==========================

* README update: example usage, pypi-version badge, Q&A.
* `ndpoly` method addition: `isconstant`, `toarray`.

Version 0.0.6 (2019-08-28)
==========================

* Rudimentary alignment of shape, indeterminants and exponents.
* Numpoly baseclass `ndpoly` with basic call functionality and interface for
  dealing with numpy interoperability.
* Numpy function addition:
  `absolute`, `add`, `any`, `all`, `array_repr`, `array_str`, `concatenate`,
  `cumsum`, `divide`, `equal`, `floor_divide`, `multiply`, `negative`,
  `not_equal`, `outer`, `positive`, `power`, `subtract`, `sum`.
* Polynomial function addition: `diff`, `gradient`, `hessian`, `to_array`,
  `to_sympy`, `to_string`, `monomial`, `symbols`.
