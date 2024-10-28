# cdispach.pyx

# Import Cython and types from standard Python
import numpy as np
cimport numpy as np
import numpoly
from typing import List


cdef List[np.ndarray] cprepare_input(
    List[np.ndarray] inputs,
    str name,
):
    cdef np.ndarray poly
    cdef List[np.ndarray] output = []

    for i in range(len(inputs)):
        poly = inputs[i]
        output.append(numpoly.cget_values(name, poly.values))

    return output


def cloop_function(
        object numpy_ufunc,
        List[np.ndarray] inputs,
        np.ndarray keys,
        np.ndarray out,
        **kwargs,
):
    cdef Py_ssize_t i
    cdef str key_str
    cdef list prepare

    for i in range(keys.shape[0]): 
        key_str = str(keys[i])
        prepare = cprepare_input(inputs, key_str)
        numpoly.cset_values(numpy_ufunc(*prepare, **kwargs), key_str, out)

