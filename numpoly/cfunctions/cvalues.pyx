# cvalues.pyx

# Import Cython and types from standard Python
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint8_t, uint32_t


cdef void cset_bool_values_1d(
        uint8_t [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef uint8_t *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <uint8_t *> (data_ptr + value_offset)
        value_ptr[0] = coeffs[i]


cdef void cset_uint32_values_1d(
        uint32_t [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef uint32_t *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <uint32_t *> (data_ptr + value_offset)
        value_ptr[0] = coeffs[i]


cdef void cset_int_values_1d(
        int64_t [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <int64_t *> (data_ptr + value_offset)
        value_ptr[0] = coeffs[i]


cdef void cset_float_values_1d(
        double [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <double *> (data_ptr + value_offset)
        value_ptr[0] = coeffs[i]


cdef void cset_complex_values_1d(
        complex [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef complex *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <complex *> (data_ptr + value_offset)
        value_ptr[0] = coeffs[i]


cpdef cset_values(
        coeffs,
        str name,
        np.ndarray out,
):
    if coeffs.dtype == np.bool:
        cset_bool_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.uint32:
        cset_uint32_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.int64:
        cset_int_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.float64:
        cset_float_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.complex128:
        cset_complex_values_1d(coeffs, name, out)
    else:
        ValueError("TYPE not implemented")


cdef void cadd_bool_values_1d(
        uint8_t [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef uint8_t *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <uint8_t *> (data_ptr + value_offset)
        value_ptr[0] += coeffs[i]


cdef void cadd_uint32_values_1d(
        uint32_t [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef uint32_t *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <uint32_t *> (data_ptr + value_offset)
        value_ptr[0] += coeffs[i]


cdef void cadd_int_values_1d(
        int64_t [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <int64_t *> (data_ptr + value_offset)
        value_ptr[0] += coeffs[i]


cdef void cadd_float_values_1d(
        double [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <double *> (data_ptr + value_offset)
        value_ptr[0] += coeffs[i]


cdef void cadd_complex_values_1d(
        complex [::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef complex *value_ptr
    
    # Size for rows (in Bytes and Int)
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    byte_row = out.strides[0]
    nrow = out.size
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <complex *> (data_ptr + value_offset)
        value_ptr[0] += coeffs[i]


cpdef cadd_values(
        coeffs,
        str name,
        np.ndarray out,
):
    if coeffs.dtype == np.bool:
        cadd_bool_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.uint32:
        cadd_uint32_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.int64:
        cadd_int_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.float64:
        cadd_float_values_1d(coeffs, name, out)
    elif coeffs.dtype == np.complex128:
        cadd_complex_values_1d(coeffs, name, out)
    else:
        ValueError("Type not implemented")
