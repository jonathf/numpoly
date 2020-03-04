"""Global configuration."""
from packaging.version import parse
import os
import pytest
import numpy

import numpoly


class MethodDispatch(object):
    def __getattr__(self, name):
        def dispatch_to_method(poly, *args, **kwargs):
            if hasattr(poly, name):
                func = getattr(poly, name)
            else:
                func = getattr(poly, "__%s__" % name)
            return func(*args, **kwargs)
        return dispatch_to_method


if parse(numpy.__version__) < parse("1.17.0"):
    # Use internal interface only (Python 2 in practice)
    INTERFACES = [numpoly, MethodDispatch()]
    FUNC_INTERFACES = [numpoly]
else:
    # use both internal and __array_function__ interface (Python 3 in practice)
    INTERFACES = [numpoly, numpy, MethodDispatch()]
    FUNC_INTERFACES = [numpoly, numpy]


@pytest.fixture(params=FUNC_INTERFACES)
def func_interface(request):
    return request.param


@pytest.fixture(params=INTERFACES)
def interface(request):
    return request.param


@pytest.fixture(autouse=True)
def global_variables(doctest_namespace, monkeypatch):
    """Ensure certain variables are available during tests."""
    doctest_namespace["numpy"] = numpy
    doctest_namespace["numpoly"] = numpoly

    environ = os.environ.copy()
    environ["NUMPOLY_DEBUG"] = True
    monkeypatch.setattr("os.environ", environ)
