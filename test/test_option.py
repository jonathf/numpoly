import pytest
import numpy
import numpoly

POLYNOMIAL_CONFIGURATIONS = [
    dict(zip(["display_graded", "display_inverse", "display_reverse", "expected_output"], args)
         ) for args in [
             (False, False, False, "1+x+x**2+y+x*y+y**2"),
             (False, True, False, "y**2+x*y+y+x**2+x+1"),
             (False, False, True, "1+y+y**2+x+x*y+x**2"),
             (False, True, True, "x**2+x*y+x+y**2+y+1"),
             (True, False, False, "1+x+y+x**2+x*y+y**2"),
             (True, True, False, "y**2+x*y+x**2+y+x+1"),
             (True, False, True, "1+y+x+y**2+x*y+x**2"),
             (True, True, True, "x**2+x*y+y**2+x+y+1"),
]]


@pytest.fixture(params=POLYNOMIAL_CONFIGURATIONS)
def display_config(request):
    yield request.param


def test_display_order(display_config):
    expected_output = display_config.pop("expected_output")
    polynomial = numpy.sum(numpoly.monomial(3, names=("x", "y")))
    with numpoly.global_options(**display_config):
        assert str(polynomial) == expected_output
