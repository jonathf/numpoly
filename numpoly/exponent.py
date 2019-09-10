from string import printable  # pylint: disable=no-name-in-module

import numpy


FORWARD_DICT = dict(enumerate(numpy.array(list(printable), dtype="S1")))
FORWARD_MAP = numpy.vectorize(FORWARD_DICT.get)
INVERSE_DICT = {value: key for key, value in FORWARD_DICT.items()}
INVERSE_MAP = numpy.vectorize(INVERSE_DICT.get)


def keys_to_exponents(keys):
    """
    >>> keys_to_exponents("0P")
    array([ 0, 51])
    >>> keys_to_exponents(["12", "21", "0P"])
    array([[ 1,  2],
           [ 2,  1],
           [ 0, 51]])
    """
    keys = numpy.asarray(keys, dtype="S")
    if not keys.shape:
        return numpy.array([INVERSE_DICT[key.encode()]
                            for key in keys.item().decode("utf-8")])
    exponents = keys.view("S1").reshape(len(keys), -1)
    return INVERSE_MAP(exponents)


def exponents_to_keys(exponents):
    """
    >>> exponents_to_keys([0, 1, 2])
    '012'
    >>> exponents_to_keys([[0, 0], [0, 1], [9, 44]])
    array(['00', '01', '9I'], dtype='<U2')
    """
    exponents = numpy.asarray(exponents, dtype=int)
    if len(exponents.shape) == 1:
        return exponents_to_keys(exponents.reshape(1, -1))[0]
    keys = FORWARD_MAP(exponents).flatten()
    keys = numpy.array(keys.view("S%d" % exponents.shape[-1]), dtype="U")
    return keys
