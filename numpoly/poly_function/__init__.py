"""Polynomial functionality."""
from .call import call
from .decompose import decompose
from .derivative import diff, gradient, hessian
from .isconstant import isconstant
from .largest_exponent import largest_exponent
from .divide import poly_divide, poly_divmod, poly_remainder
from .set_dimensions import set_dimensions
from .sortable_proxy import sortable_proxy
from .tonumpy import tonumpy

__all__ = ("call", "decompose", "diff", "gradient", "hessian", "isconstant",
           "largest_exponent", "poly_divide", "poly_divmod", "poly_remainder",
           "set_dimensions", "sortable_proxy", "tonumpy")
