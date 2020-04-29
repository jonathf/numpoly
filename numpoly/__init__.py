# pylint: disable=wildcard-import
"""Numpoly -- Multivariate polynomials as numpy elements."""
import logging
import os
import pkg_resources

from .baseclass import ndpoly

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
)
from .sympy_ import to_sympy

from .array_function import *
from .poly_function import *

try:
    __version__ = pkg_resources.get_distribution("numpoly").version
except pkg_resources.DistributionNotFound:
    __version__ = None


def configure_logging():
    """Configure logging for Numpoly."""
    logpath = os.environ.get("NUMPOLY_LOGPATH", os.devnull)
    logging.basicConfig(level=logging.DEBUG, filename=logpath, filemode="w")
    streamer = logging.StreamHandler()
    loglevel = logging.DEBUG if os.environ.get("NUMPOLY_DEBUG", "") else logging.WARNING
    streamer.setLevel(loglevel)

    logger = logging.getLogger(__name__)
    logger.addHandler(streamer)

configure_logging()
