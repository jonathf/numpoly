"""Global numpoly options."""
from contextlib import contextmanager

GLOBAL_OPTIONS_DEFAULTS = {
    "default_varname": "q",
    "force_number_suffix": True,
    "varname_filter": r"q\d+",
    "sort_graded": True,
    "sort_reverse": False,
    "display_graded": True,
    "display_reverse": False,
    "display_inverse": True,
    "display_exponent": "**",
    "display_multiply": "*",
}
_NUMPOLY_OPTIONS = GLOBAL_OPTIONS_DEFAULTS.copy()


def get_options(defaults=False):
    """
    Get global numpoly options.

    Args:
        defaults (bool):
            Return the options for the global option defaults, instead of the
            options currently in use.

    Returns:
        (Dict[str, Any]):
            Collection of values that affect the numpoly machinery.
            See `numpoly.set_options` for details.

    Examples:
        >>> options = get_options()
        >>> options["default_varname"]
        'q'

    See also:
        `set_options`, `global_options`

    """
    if defaults:
        return GLOBAL_OPTIONS_DEFAULTS.copy()
    return _NUMPOLY_OPTIONS.copy()


def set_options(**kwargs):
    """
    Set global numpoly options.

    Args:
        default_varname (str):
            Polynomial indeterminant defaults, if not defined explicitly.
        force_number_suffix (bool):
            Add a postfix index to single indeterminant name. If
            single indeterminant name, e.g. 'q' is provided, but the
            polynomial is multivariate, an extra postfix index is
            added to differentiate the names: 'q0, q1, q2, ...'. If
            true, enforce this behavior for single variables as well
            such that 'q' always get converted to 'q0'.
        varname_filter (str):
            Regular expression defining valid indeterminant names.
        sort_graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``x**2*y**2*z**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``x**3*y*z``, ``x*y**2*z and
            ``x*y*z**2``, which all have exponent sum of 5.
        sort_reverse (bool):
            Reverse lexicographical sorting meaning that ``x*y**3`` is
            considered bigger than ``x**3*y``, instead of the opposite.
        display_graded (bool):
            When displaying polynomials as strings, sort polynomial sums in graded order.
        display_reverse (bool):
            When displaying polynomials as strings, sort polynomial sums in reversed lexicographical order.
        display_inverse (bool):
            If true, display polynomials from smallest to largest exponent.
        display_exponent (str):
            Exponent sign; Separate indeterminants and its power.
        display_multiply (str):
            Multiplication sign; Separates coefficients and
            indeterminants, and indeterminants from each other.

    Examples:
        >>> numpoly.monomial([3, 3])
        polynomial([1, q0, q0**2, q1, q0*q1, q1**2])
        >>> numpoly.set_options(default_varname="z", varname_filter=".+")
        >>> numpoly.monomial([3, 3])
        polynomial([1, z0, z0**2, z1, z0*z1, z1**2])

    See also:
        `get_options`, `global_options`

    """
    for key in kwargs:
        if key not in _NUMPOLY_OPTIONS:
            raise KeyError("option '%s' not recognised." % key)
    _NUMPOLY_OPTIONS.update(**kwargs)


@contextmanager
def global_options(**kwargs):
    """
    Temporarily set global numpoly options.

    Args:
        kwargs (Any):
            Collection of values that deviate from the defaults.
            See `numpoly.set_options` for details.

    Yields:
        (Dict[str, Any]):
            Collection of values that affect the numpoly machinery.
            See `numpoly.set_options` for details.

    Examples:
        >>> numpoly.get_options()["default_varname"]
        'q'
        >>> with numpoly.global_options(default_varname="X"):
        ...     print(numpoly.get_options()["default_varname"])
        X
        >>> numpoly.get_options()["default_varname"]
        'q'

    See also:
        `get_options`, `set_options`

    """
    options = get_options()
    try:
        set_options(**kwargs)
        yield get_options()
    finally:
        set_options(**options)
