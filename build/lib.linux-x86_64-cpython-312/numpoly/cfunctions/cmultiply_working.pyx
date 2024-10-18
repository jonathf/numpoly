# cmultiply.pyx

# Import Cython and types from standard Python
import numpy as np
cimport numpy as np
from libc.stdlib cimport free
from libc.stdio cimport sprintf
from cpython cimport bool as cbool

ctypedef fused int_or_array1:
    list[np.int64_t] 
    np.ndarray[np.float64_t, ndim=2] 

ctypedef fused int_or_array2:
    list[np.int64_t] 
    np.ndarray[np.float64_t, ndim=2] 

cdef int find_index(tuple names, str key):
    cdef int i, n
    n = len(names)
    
    for i in range(n):
        if names[i] == key:
            return i
    
    return -1

cdef np.ndarray[np.float64_t, ndim=2] cmultiply_cdef(
        np.ndarray[np.uint32_t, ndim=2] expons1, 
        np.ndarray[np.uint32_t, ndim=2] expons2,
        int_or_array1 coeffs1, 
        int_or_array2 coeffs2,
        int offset,
        tuple names,
        np.ndarray[np.float64_t, ndim=2] values
):
    cdef set seen = set()
    cdef int i, j, k, idx
    cdef char key[256]
    cdef int key_len
    cdef str key_str

    for i in range(expons1.shape[0]):
        for j in range(expons2.shape[0]):
            key_len = 0

            for k in range(expons1.shape[1]):
                key_len += sprintf(key + key_len, "%c", expons1[i, k] + expons2[j, k] + offset)

            key_str = key[:key_len].decode('utf-8')
            idx = find_index(names, key_str)
            if idx == -1:
                raise KeyError(f"Key {key_str} not found in names")

            if key_str in seen:
                values[idx] += coeffs1[i] * coeffs2[j]
            else:
                values[idx] = coeffs1[i] * coeffs2[j]
                seen.add(key_str)

    return values

def cmultiply(
        np.ndarray[np.uint32_t, ndim=2] expons1, 
        np.ndarray[np.uint32_t, ndim=2] expons2,
        int_or_array1 coeffs1, 
        int_or_array2 coeffs2,
        int offset,
        tuple names,
        np.ndarray[np.float64_t, ndim=2] values
):
    return cmultiply_cdef(expons1, expons2, coeffs1, coeffs2, offset, names, values)
