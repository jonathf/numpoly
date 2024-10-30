# cmultiply.pyx

# Import Cython and types from standard Python
import numpy as np
cimport numpy as np
from libc.stdio cimport sprintf
import numpoly
from typing import List


cdef void cmultiply_cdef(
        np.ndarray[np.uint32_t, ndim=2] expons1, 
        np.ndarray[np.uint32_t, ndim=2] expons2,
        List[np.ndarray] coeffs1, 
        List[np.ndarray] coeffs2,
        int offset,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, k

    # Declare variables for string representation of polynomials
    cdef set seen = set()
    cdef char key[256]
    cdef Py_ssize_t key_len
    cdef str key_str

    for i in range(expons1.shape[0]):
        for j in range(expons2.shape[0]):
            key_len = 0
            for k in range(expons1.shape[1]):
                key_len += sprintf(key + key_len, "%c", expons1[i, k] + expons2[j, k] + offset)
            
            key_str = key[:key_len].decode('utf-8')
            if key_str in seen:
                numpoly.cadd_values((coeffs1[i] * coeffs2[j]).ravel(), key_str, out)
            else:
                numpoly.cset_values((coeffs1[i] * coeffs2[j]).ravel(), key_str, out)
                seen.add(key_str)


def cmultiply(
        np.ndarray[np.uint32_t, ndim=2] expons1, 
        np.ndarray[np.uint32_t, ndim=2] expons2,
        List[np.ndarray] coeffs1, 
        List[np.ndarray] coeffs2,
        int offset,
        np.ndarray out,
):
    cmultiply_cdef(expons1, expons2, coeffs1, coeffs2, offset, out)
