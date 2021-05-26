import numpy
from numpoly import polynomial
import numpoly


X, Y = numpoly.variable(2)


def test_align_polynomials():
    poly1 = 5+3*Y+3*X
    poly2 = 4.
    poly1_, poly2_ = numpoly.align_polynomials(poly1, poly2)
    assert poly1 == poly1_
    assert poly2 == poly2_
    assert numpy.all(poly1_.exponents == poly2_.exponents)
    assert poly1_.exponents.shape[-1] == 2
    assert poly1_.shape == poly2_.shape

    X_, Y_ = numpoly.align_polynomials(X, Y)
    assert not X_.shape
    assert not Y_.shape
    assert X_ == X
    assert Y_ == Y
