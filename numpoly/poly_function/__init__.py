"""Polynomial functionality."""
from .call import call
from .decompose import decompose
from .derivative import derivative, gradient, hessian
from .isconstant import isconstant
from .lead_coefficient import lead_coefficient
from .lead_exponent import lead_exponent
from .divide import (
    get_division_candidate, poly_divide, poly_divmod, poly_remainder)
from .set_dimensions import set_dimensions
from .sortable_proxy import sortable_proxy
from .tonumpy import tonumpy

__all__ = (
    "call", "decompose", "derivative", "gradient", "hessian", "isconstant",
    "lead_coefficient", "lead_exponent", "poly_divide", "poly_divmod",
    "poly_remainder", "set_dimensions", "sortable_proxy", "tonumpy",
)
