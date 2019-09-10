""""""
import numpy
import numpoly

from .implements import implements


@implements(numpy.power)
def power(x1, x2, **kwargs):
    x1 = x1.copy()
    x2 = numpy.asarray(x2, dtype=int)

    if not x2.shape:
        out = numpoly.polynomial_from_attributes(
            [(0,)], [numpy.ones(x1.shape, dtype=x1._dtype)], x1._indeterminants[:1])
        for _ in range(x2.item()):
            out = numpoly.multiply(out, x1, **kwargs)

    elif x1.shape:
        if x2.shape[-1] == 1:
            if x1.shape[-1] == 1:
                out = numpoly.power(x1.T[0].T, x2.T[0].T).T[numpy.newaxis].T
            else:
                out = numpoly.concatenate([power(x, x2.T[0])[numpy.newaxis] for x in x1.T], axis=0).T
        elif x1.shape[-1] == 1:
            out = numpoly.concatenate([power(x1.T[0].T, x.T).T[numpy.newaxis] for x in x2.T], axis=0).T
        else:
            out = numpoly.concatenate([power(x1_, x2_).T[numpy.newaxis] for x1_, x2_ in zip(x1.T, x2.T)], axis=0).T
    else:
        out = numpoly.concatenate([power(x1, x.T).T[numpy.newaxis] for x in x2.T], axis=0).T
    return out

