# cfrom_attributes.pyx

# Import Cython and types from standard Python
import numpy as np
cimport numpy as np
import numpoly
from typing import List


def cfrom_attributes(
    List[np.ndarray] coeffs,
    np.ndarray poly
):
    cdef Py_ssize_t i, nfields

    # Declare variables for string representation of polynomials
    names = poly.dtype.names
    nfields = len(names)

    for i in range(nfields):
        name = str(names[i])
        numpoly.cset_values(coeffs[i].ravel(), name, poly)

