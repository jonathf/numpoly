import pytest
import numpy
import numpoly

POLYNOMIAL_CONFIGURATIONS = [
    dict(zip(["display_graded", "display_inverse", "display_reverse", "expected_output"], args)
         ) for args in [
             (False, False, False, "1+q0+q0**2+q1+q0*q1+q1**2"),
             (False, True, False, "q1**2+q0*q1+q1+q0**2+q0+1"),
             (False, False, True, "1+q1+q1**2+q0+q0*q1+q0**2"),
             (False, True, True, "q0**2+q0*q1+q0+q1**2+q1+1"),
             (True, False, False, "1+q0+q1+q0**2+q0*q1+q1**2"),
             (True, True, False, "q1**2+q0*q1+q0**2+q1+q0+1"),
             (True, False, True, "1+q1+q0+q1**2+q0*q1+q0**2"),
             (True, True, True, "q0**2+q0*q1+q1**2+q0+q1+1"),
]]


@pytest.fixture(params=POLYNOMIAL_CONFIGURATIONS)
def display_config(request):
    """Parameterization of various display options."""
    yield request.param


def test_display_order(display_config):
    """Ensure string output changes with various display options."""
    expected_output = display_config.pop("expected_output")
    polynomial = numpy.sum(numpoly.monomial(3, dimensions=("q0", "q1")))
    with numpoly.global_options(**display_config):
        assert str(polynomial) == expected_output


def test_illegal_option():
    """Ensure that illegal arguments raises error."""
    with pytest.raises(KeyError):
        numpoly.set_options(not_an_argument=4)
    with pytest.raises(KeyError):
        with numpoly.global_options(not_an_argument=4):
            pass
