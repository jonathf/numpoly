Master Branch
=============

Version 1.2.4 (2022-11-16)
==========================

Bug and pipeline fixes.

CHANGED:
  * Update CircleCI to use pip instead of poetry.
FIXED:
  * Polynomial alignment for more than 10 variables.

Version 1.2.3 (2021-05-26)
==========================

Reduce the number of alignments.

ADDED:
  * Reduce the number memory copies, increasing speed for large arrays.

Version 1.2.2 (2021-05-23)
==========================

ADDED:
  * Removed redundant calculations from multiplications, drastically improving
    computational cost in some cases.

Version 1.2.1 (2021-05-23)
==========================

FIXED:
  * removed redundant and six imports

Version 1.2.0 (2021-05-21)
==========================

ADDED:
  * Annotations everywhere!
  * User provided type `PolyLike`: polynmial analog to `ArrayLike`.
  * CI: mypy checks.

CHANGED:
  * CI: Replaced workspace attaching in favor of independent runs.
  * CI: Linting, pydocstyle, Sphinx, mypy in py39, and pytest and coverage in py37.

REMOVED:
  * Support for python versions 2 and < 3.7.

Version 1.1.4 (2021-05-18)
==========================

FIXED:
  * truncate function affected by rounding error.

Version 1.1.3 (2021-04-02)
==========================

ADDED:
  * Numpy function addition: `det` (from `np.linalg`).

Version 1.1.2 (2021-03-29)
==========================

ADDED:
  * Numpy function addition: `roots`.

Version 1.1.1 (2020-12-04)
==========================

ADDED:
  * Functions `remove_redundant_coefficients` and
    `remove_redundant_names` to explicitly remove junk information.
  * Global flags `retain_coefficients` and `retain_names` to set global
    defaults.

CHANGED:
  * Post-process with `postprocess_attribute` mandatory for all calls to
    `clean_attributes`, but because of the new flags, keeping polynomial
    unchanged is still possible.

REMOVED:
  * Flag `clean` arg replaced with more explicit `retain_coefficients` and
    `retain_names`.

Version 1.1.0 (2020-11-19)
==========================

CHANGED:
  * The call signature of `numpoly.call` changed to make its usage different
    from `__call__`.
  * the call signature of `numpoly.monomial` change to make it more aligned
    with its parent `numpoly.glexindex`.

Version 1.0.9 (2020-11-18)
==========================

ADDED:
  * Numpy function additions: `one_like`, `zeros_like`.

CHANGED:
  * Enforce 100 percent coverage in CI.

Version 1.0.8 (2020-10-29)
==========================

ADDED:
  * Numpy function additions: `ediff1d`, `full`, `full_like`, `copyto`.

Version 1.0.7 (2020-10-28)
==========================

ADDED:
  * Polynomial constructor: `polynomial_from_roots`.

Version 1.0.6 (2020-07-27)
==========================

ADDED:
  * Tests for `numpoly.equal`.
  * Numpy function additions: `result_type`, `diff`.

CHANGED:
  * First iteration for deprecating `align_shape`:
    * Added common `assert_equal` function to test contiguous-ness, shape,
      dtype etc. in a more automated way.
    * Small patches here and there to assert contiguous-ness.
  * Rename single variable differentiation:
    `diff -> derivative` (as the former is reserved in numpy).

Version 1.0.5 (2020-07-08)
==========================

ADDED:
  * Pickle support.
  * Polynomial function: `lead_coefficient`.
  * (Unwrapped) numpy functions: `load`, `loadtxt`.
  * Numpy function additions: `save`, `savetxt`, `savez`, `savez_compressed`.

CHANGED:
  * Rename function `largest_exponent -> lead_exponent`.

FIXED:
  * Bugfix: Poly-division with large relative error caused infinity-loops.

Version 1.0.4 (2020-07-01)
==========================

ADDED:
  * Numpy function additions: `diag`, `diagonal`, `ones`
  * Added changelog (the file you currently are reading).

Version 1.0.3 (2020-06-26)
==========================

FIXED:
  * Bugfix in `set_dimensions` (1.0.2 solution didn't work).

Version 1.0.2 (2020-06-26)
==========================

FIXED:
  * Bugfix in `set_dimensions`.

Version 1.0.1 (2020-06-26)
==========================

ADDED:
  * Polynomial function addition: `variable`, `set_dimensions`.

Version 1.0.0 (2020-06-26)
==========================

CHANGED:
  * Enforce all polynomials on the format `"q\d+"` aligning the defaults with
    chaospy.
  * Lots and lots of doctests updated.

Version 0.3.0 (2020-06-08)
==========================

ADDED:
  * Comparison operator support!
  * Polynomial functions:
    `poly_divide`, `poly_divmod`, `poly_reminder`, `sortable_proxy`.
  * Numpy functions:
    `amax`, `amin`, `argmin`. `argmax`, `greater`, `greater_equal`,
    `less`, `less_equal`, `max`, `maximum`, `min`, `minimum`.
  * Lots of extra tests.

CHANGED
  * String representation changed from insertion order to Graded-reverse
    lexicographically.
  * New comparison and poly-division and derivative chapter in Sphinx, instead
    of poly-functions.
  * Move global options into common file with interfaces:
    `get_options`, `set_options` and `global_options` (context manager).

Version 0.2.3 (2020-05-08)
==========================

CHANGED:
  * More aggressive cross-truncation approach to keep memory low under
    `bindex`.

Version 0.2.2 (2020-05-08)
==========================

CHANGED:
  * More documentation updates.
  * Cleanup to `bindex` for better handle of implicit dimensions.

Version 0.2.1 (2020-05-06)
==========================

CHANGED:
  * Documentation updates.

Version 0.2.0 (2020-05-05)
==========================

ADDED:
  * Polynomial division support!
    * Polynomial functions: `poly_divide`, `poly_divmod`, `poly_reminder`.
    * Numpy functions:
    `true_divide`, `divmod`, `remainder`. `where`, `zeros`.

Version 0.1.17 (2020-04-29)
===========================

ADDED:
  * Numpy function additions:
    `apply_along_axis`, `apply_over_axes`, `expand_dims`.

Version 0.1.16 (2020-04-18)
===========================

FIXED:
  * Bugfix to `bindex` in handle of indices around 0.

Version 0.1.15 (2020-04-17)
===========================

ADDED:
  * Support for enforced naming convention (for use in Chaospy).

Version 0.1.14 (2020-04-15)
===========================

CHANGED:
  * Refactor of `monomial` again to deal with speed issue.

Version 0.1.13 (2020-03-31)
===========================

ADDED:
  * Allow for debugging messages through `$NUMPOLY_DEBUG` environmental
    variable.
  * Numpy functions: `count_nonzero`, `nonzero` (thanks Fredrik Meyer)
  * Package version number added to `numpoly.__version__`.

CHANGED:
  * Better py2 support.

REMOVED:
  * Remove CircleCI cache for py2 (as it is really light).

Version 0.1.12 (2020-03-02)
===========================

FIXED:
  * Bugfix for edge case in `prod`.

Version 0.1.11 (2020-02-26)
===========================

ADDED:
  * Support for the "empty set" polynomial: `polynomial([])`.

Version 0.1.10 (2020-02-26)
===========================

CHANGED:
  * Refactor `monomial`, cleaning it out and catching some subtle bugs.

Version 0.1.9 (2020-02-26)
==========================

CHANGED:
  * Documentation cleanup.

FIXED:
  * Small bugfix in `monomial` in how it implicitly handles multiple
    dimensions.

Version 0.1.8 (2020-02-24)
==========================

ADDED:
  * Numpy function additions: `matmul`.

Version 0.1.7 (2020-02-11)
==========================

ADDED:
  * Numpy function additions: `broadcast_arrays`.

Version 0.1.6 (2020-01-10)
==========================

FIXED:
  * Small bugfix in experimental code.

Version 0.1.5 (2020-01-10)
==========================

ADDED:
  * Numpy functions: `array_split`, `dsplit`, `hsplit`, `split`, `vsplit`.

CHANGED:
  * Documentation update.

FIXED:
  * Move key index offset from 48 (the visually appealing 0, 1, 2, ...)
    to 59 (skipping the problematic 58 ':').

Version 0.1.4 (2019-12-01)
==========================

FIXED:
  * Numpy function: `tile` (sourced not added in 0.1.3).

Version 0.1.3 (2019-12-01)
==========================

ADDED:
  * More documentation.
  * Numpy functions: `transpose`, `tile`.

CHANGED:
  * Rename function arg `{indeterminants -> names}` (all over the place).

Version 0.1.2 (2019-11-26)
==========================

ADDED:
  * CI tests for the  root readme.
  * New Numpoly logo.
  * Introduction chapter added to Sphinx.
  * Numpy functions: `choose`, `reshape`.

CHANGED:
  * Collection of global constant moved into common dictionary.

Version 0.1.1 (2019-11-21)
==========================

CHANGED:
  * Small documentation update.

Version 0.1.0 (2019-11-17)
==========================

ADDED:
  * Support for alpha, beta, rc, dev, post releases.
  * Validate tags against install version.
  * Polynomial function: `decompose`.

Version 0.0.17 (2019-10-20)
===========================

CHANGED:
  * Change string representation to display polynomial
    by insertion order (affecting a lot of examples).

Version 0.0.16 (2019-10-01)
===========================

ADDED:
  * Global constants added for manipulating string representation.

CHANGED:
  * `monomial`: reorder args such that `indeterminants` argument is at the end.

Version 0.0.15 (2019-09-27)
===========================

ADDED:
  * Include Sphinx docs in CircleCI testing.
  * Add Construct chapter to Sphinx.
  * Numpy function additions: `repeat`.

CHANGED:
  * Rename `toarray -> tonumpy`, `as_ndarray -> values` (function -> property).

Version 0.0.14 (2019-09-27)
===========================

ADDED:
  * Documentation update:
    * Introduction to `ndpoly` baseclass added.
    * Polynomial function collection.
    * Enforce complete function list through `sphinx_automodapi`.
    * Read-the-docs deployment configuration.
  * `ndpoly` method: `as_ndarray`.
  * Numpy function: `stack`.

CHANGED:
  * Remove functions mappings between exponents and keys in favor of in-line
    solution.

FIXED:
  * Bugfixes and code cleanups for `concatenate`, `*stack` and multiplications.

Version 0.0.13 (2019-09-25)
===========================

ADDED:
  * First iteration Sphinx docs.
  * Increased testing coverage.
  * Numpy function additions: `atleast_1d`, `atleast_2d`, `atleast_3d`,
    `ceil`, `floor`, `dstack`, `hstack`, `vstack`.

CHANGED:
  * CircleCI cleanup: limited py27 and full py37 testing only.

Version 0.0.12 (2019-09-13)
===========================

ADDED:
  * Add align_dtype to alignment process.

CHANGED:
  * Recast dtype support in `ndpoly.__call__` when input is other format than
    internal one.

Version 0.0.11 (2019-09-12)
===========================

ADDED:
  * Numpy functions: `prod`, `moveaxis`.

CHANGED:
  * Move testing dispatching to `conftest.py`.
  * Testing polish.

Version 0.0.10 (2019-09-12)
===========================

ADDED:
  * Testing of alignment.
  * Numpy function additions: `allclose`, `isclose`, `isfinite`, `mean`.

CHANGED:
  * Split testing suite into py2 and py3
    (as py3 supports full dispatching, and py2 does not).

FIXED:
  * Variable name typo fixes.

Version 0.0.9 (2019-09-12)
==========================

ADDED:
  * Linting to CircleCI checks.
  * `simple_dispatch` function to unify the backend for the most simplest
    numpy functions.
  * Support for numpy reduce and accumulate mappings.
  * `ndpoly` methods: `from_attributes`,
    `round` (likely needed because of numpy bug).
  * Numpy functions: `logical_and`, `rind`, `square`.

CHANGED:
  * Some code clean-up of alignment.
  * Refactor constructions functions.
  * Renamings: `ndpoly.{_exponents -> keys}`,
    `ndpoly.{_indeterminants -> names}`,
    `numpy.{clean_polynomial_attributes -> clean_attributes}`

Version 0.0.8 (2019-09-11)
==========================

ADDED:
  * Polynomial functions: `aspolynomial`.
  * Numpy functions: `around`, `common_type`, `inner`, `logical_or`.

CHANGED:
  * Functions for mapping between `Tuple[int, ...]` and `str` for dealing with
    exponents, instead of using exposed maps.
  * Split array functions into one-file-per-function.

Version 0.0.7 (2019-09-08)
==========================

ADDED:
  * README with example usage, pypi-version badge, Q&A.
  * `ndpoly` methods: `isconstant`, `toarray`.

Version 0.0.6 (2019-08-28)
==========================

ADDED:
  * Rudimentary alignment of shape, indeterminants and exponents.
  * Numpoly baseclass `ndpoly` with basic call functionality and interface
    for dealing with numpy interoperability.
  * Numpy functions: `absolute`, `add`, `any`, `all`, `array_repr`,
    `array_str`, `concatenate`, `cumsum`, `divide`, `equal`, `floor_divide`,
    `multiply`, `negative`, `not_equal`, `outer`, `positive`, `power`,
    `subtract`, `sum`.
  * Polynomial functions: `diff`, `gradient`, `hessian`, `to_array`,
    `to_sympy`, `to_string`, `monomial`, `symbols`.
