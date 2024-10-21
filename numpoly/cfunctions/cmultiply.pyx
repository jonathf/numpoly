# cmultiply.pyx

# Import Cython and types from standard Python
import numpy as np
cimport numpy as np
from libc.stdlib cimport free
from libc.stdio cimport sprintf
from cpython cimport bool as cbool


cdef void cmultiply_cdef(
        np.ndarray[np.uint32_t, ndim=2] expons1, 
        np.ndarray[np.uint32_t, ndim=2] expons2,
        np.ndarray[np.float64_t, ndim=2] coeffs1, 
        np.ndarray[np.float64_t, ndim=2] coeffs2,
        int offset,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, k, idx, byte_row, nrow

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr

    # Declare variables for string representation of polynomials
    cdef set seen = set()
    cdef char key[256]
    cdef Py_ssize_t key_len
    cdef str key_str
    cdef Py_ssize_t value_offset

    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    nrow = out.shape[0]

    for i in range(expons1.shape[0]):
        for j in range(expons2.shape[0]):
            key_len = 0

            for k in range(expons1.shape[1]):
                key_len += sprintf(key + key_len, "%c", expons1[i, k] + expons2[j, k] + offset)
            
            key_str = key[:key_len].decode('utf-8')
            value_offset = <Py_ssize_t> out.dtype.fields[key_str][1]
            if key_str in seen:
                for k in range(nrow):
                    # Get the pointers to the specific memory location
                    data_ptr = base_ptr + k * byte_row

                    # Access fields with offsets
                    value_ptr = <double *> (data_ptr + value_offset)
                    value_ptr[0] += coeffs1[i, k] * coeffs2[j, k]

            else:
                for k in range(nrow):
                    # Get the pointers to the specific memory location
                    data_ptr = base_ptr + k * byte_row

                    # Access fields with offsets
                    value_ptr = <double *> (data_ptr + value_offset)
                    value_ptr[0] = coeffs1[i, k] * coeffs2[j, k]
                seen.add(key_str)

def cmultiply(
        np.ndarray[np.uint32_t, ndim=2] expons1, 
        np.ndarray[np.uint32_t, ndim=2] expons2,
        np.ndarray[np.float64_t, ndim=2] coeffs1, 
        np.ndarray[np.float64_t, ndim=2] coeffs2,
        int offset,
        np.ndarray out,
):
    cmultiply_cdef(expons1, expons2, coeffs1, coeffs2, offset, out)
