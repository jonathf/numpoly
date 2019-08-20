"""Global configuration."""
import pytest


@pytest.fixture(autouse=True)
def doctest_variables(doctest_namespace):
    """Ensure certain variables are available during doctests."""
    import numpy
    doctest_namespace["numpy"] = numpy
    import numpoly
    doctest_namespace["numpoly"] = numpoly
