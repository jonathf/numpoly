# pylint: disable=wildcard-import
"""Numpoly -- Multivariate polynomials as numpy elements."""
import logging
import os
import pkg_resources

from .baseclass import ndpoly, FeatureNotSupported

from .align import (
    align_polynomials,
    align_exponents,
    align_indeterminants,
    align_shape,
    align_dtype,
)
from .construct import (
    polynomial,
    aspolynomial,
    clean_attributes,
    remove_redundant_coefficients,
    remove_redundant_names,
    polynomial_from_attributes,
    polynomial_from_roots,
    monomial,
    symbols,
    variable,
)
from .sympy_ import to_sympy

from .array_function import *
from .poly_function import *
from .utils import (
    bindex,
    cross_truncate,
    glexindex,
    glexsort,
)
from .option import get_options, set_options, global_options


def get_version(name):
    """
    Get distribution version number, if it exists.

    Examples:
        >>> get_version("numpy") is None
        False
        >>> get_version("not_an_distribution") is None
        True

    """
    version = None
    try:
        version = pkg_resources.get_distribution(name).version
    except pkg_resources.DistributionNotFound:
        pass
    return version


def configure_logging():
    """Configure logging for Numpoly."""
    logpath = os.environ.get("NUMPOLY_LOGPATH", os.devnull)
    logging.basicConfig(level=logging.DEBUG, filename=logpath, filemode="w")
    streamer = logging.StreamHandler()
    loglevel = logging.DEBUG if os.environ.get("NUMPOLY_DEBUG", "") else logging.WARNING
    streamer.setLevel(loglevel)

    logger = logging.getLogger(__name__)
    logger.addHandler(streamer)


__version__ = get_version("numpoly")
configure_logging()
