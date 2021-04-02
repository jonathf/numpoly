"""Global configuration."""
import os
from packaging.version import parse
import pytest
import numpy

import numpoly


COLLECTION = {
    "divide": "__truediv__",
    "true_divide": "__truediv__",
    "floor_divide": "__floordiv__",
    "greater": "__gt__",
    "greater_equal": "__ge__",
    "less": "__lt__",
    "less_equal": "__le__",
    "equal": "__eq__",
    "not_equal": "__ne__",
}


class MethodDispatch(object):
    def __getattr__(self, name):
        def dispatch_to_method(poly, *args, **kwargs):
            if hasattr(poly, name):
                func = getattr(poly, name)
            elif name in COLLECTION:
                func = getattr(poly, COLLECTION[name])
            else:
                func = getattr(poly, "__%s__" % name)
            return func(*args, **kwargs)
        return dispatch_to_method
    def __repr__(self):
        return "method"


INTERFACES = ["numpoly", "numpy", "method"]
FUNC_INTERFACES = ["numpoly", "numpy"]
ALL_INTERFACES = {"numpy": numpy, "numpoly": numpoly, "method": MethodDispatch()}


@pytest.fixture(params=FUNC_INTERFACES)
def func_interface(request):
    return ALL_INTERFACES[request.param]


@pytest.fixture(params=INTERFACES)
def interface(request):
    return ALL_INTERFACES[request.param]


@pytest.fixture(scope="function", autouse=True)
def global_variables(doctest_namespace, monkeypatch):
    """Ensure certain variables are available during tests."""
    doctest_namespace["numpy"] = numpy
    doctest_namespace["numpoly"] = numpoly

    environ = os.environ.copy()
    environ["NUMPOLY_DEBUG"] = True
    monkeypatch.setattr("os.environ", environ)

    with numpoly.global_options(**numpoly.get_options(defaults=True)):
        yield
