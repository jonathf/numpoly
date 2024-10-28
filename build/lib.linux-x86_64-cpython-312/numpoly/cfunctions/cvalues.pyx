# cvalues.pyx

# Import Cython and types from standard Python
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t


cdef void cset_int_values_0d(
        int64_t coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr

    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    data_ptr = base_ptr

    value_ptr = <int64_t *> (data_ptr + value_offset)
    value_ptr[0] = coeffs


cdef void cset_float_values_0d(
        double coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr

    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    data_ptr = base_ptr

    value_ptr = <double *> (data_ptr + value_offset)
    value_ptr[0] = coeffs


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
    byte_row = out.strides[0]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
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
    byte_row = out.strides[0]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <double *> (data_ptr + value_offset)
        value_ptr[0] = coeffs[i]


cdef void cset_int_values_2d(
        int64_t [:, ::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, byte_row, byte_col, nrow, ncol, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr

    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    byte_col = out.strides[1]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    ncol = out.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            data_ptr = base_ptr + i * byte_row + j * byte_col
            value_ptr = <int64_t *> (data_ptr + value_offset)
            value_ptr[0] = coeffs[i, j]


cdef void cset_float_values_2d(
        double [:, ::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, byte_row, byte_col, nrow, ncol, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr

    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    byte_col = out.strides[1]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    ncol = out.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            data_ptr = base_ptr + i * byte_row + j * byte_col
            value_ptr = <double *> (data_ptr + value_offset)
            value_ptr[0] = coeffs[i, j]


cpdef cset_values(
        coeffs,
        str name,
        np.ndarray out,
):
    if isinstance(coeffs, np.ndarray):
        if coeffs.ndim == 1 and coeffs.dtype == np.int64:
            cset_int_values_1d(coeffs, name, out)
        elif coeffs.ndim == 1 and coeffs.dtype == np.float64:
            cset_float_values_1d(coeffs, name, out)
        elif coeffs.ndim == 2 and coeffs.dtype == np.int64:
            cset_int_values_2d(coeffs, name, out)
        elif coeffs.ndim == 2 and coeffs.dtype == np.float64:
            cset_float_values_2d(coeffs, name, out)
        else:
            ValueError("Dimensions of 'coeffs' argument must be 1 or 2")
    else:
        if isinstance(coeffs, np.int64):
            cset_int_values_0d(coeffs, name, out)
        else:
            cset_float_values_0d(coeffs, name, out)


cdef void cadd_int_values_0d(
        int64_t coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr

    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    data_ptr = base_ptr

    value_ptr = <int64_t *> (data_ptr + value_offset)
    value_ptr[0] += coeffs


cdef void cadd_float_values_0d(
        double coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr

    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    data_ptr = base_ptr

    value_ptr = <double *> (data_ptr + value_offset)
    value_ptr[0] += coeffs


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
    byte_row = out.strides[0]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
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
    byte_row = out.strides[0]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <double *> (data_ptr + value_offset)
        value_ptr[0] += coeffs[i]


cdef void cadd_int_values_2d(
        int64_t [:, ::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, byte_row, byte_col, nrow, ncol, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr

    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    byte_col = out.strides[1]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    ncol = out.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            data_ptr = base_ptr + i * byte_row + j * byte_col
            value_ptr = <int64_t *> (data_ptr + value_offset)
            value_ptr[0] += coeffs[i, j]


cdef void cadd_float_values_2d(
        double [:, ::1] coeffs, 
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, byte_row, byte_col, nrow, ncol, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr

    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    byte_col = out.strides[1]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    ncol = out.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            data_ptr = base_ptr + i * byte_row + j * byte_col
            value_ptr = <double *> (data_ptr + value_offset)
            value_ptr[0] += coeffs[i, j]


cpdef cadd_values(
        coeffs,
        str name,
        np.ndarray out,
):
    if isinstance(coeffs, np.ndarray):
        if coeffs.ndim == 1 and coeffs.dtype == np.int64:
            cadd_int_values_1d(coeffs, name, out)
        elif coeffs.ndim == 1 and coeffs.dtype == np.float64:
            cadd_float_values_1d(coeffs, name, out)
        elif coeffs.ndim == 2 and coeffs.dtype == np.int64:
            cadd_int_values_2d(coeffs, name, out)
        elif coeffs.ndim == 2 and coeffs.dtype == np.float64:
            cadd_float_values_2d(coeffs, name, out)
        else:
            ValueError("Dimensions of 'coeffs' argument must be 1 or 2")
    else:
        if isinstance(coeffs, np.int64):
            cadd_int_values_0d(coeffs, name, out)
        else:
            cadd_float_values_0d(coeffs, name, out)


cdef int64_t cget_int_values_0d(
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr

    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    data_ptr = base_ptr

    value_ptr = <int64_t *> (data_ptr + value_offset)
    return value_ptr[0]


cdef double cget_float_values_0d(
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr

    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    data_ptr = base_ptr

    value_ptr = <double *> (data_ptr + value_offset)
    return value_ptr[0]


cdef np.ndarray[np.int64_t, ndim=1] cget_int_values_1d(
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr
    
    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    cdef int64_t [::1] value = np.empty(nrow, dtype=np.int64)
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <int64_t *> (data_ptr + value_offset)
        value[i] = value_ptr[0]

    return np.asarray(value)


cdef np.ndarray[np.float64_t, ndim=1] cget_float_values_1d(
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, byte_row, nrow, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr
    
    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    cdef double [::1] value = np.empty(nrow, dtype=np.float64)
    for i in range(nrow):
        data_ptr = base_ptr + i * byte_row
        value_ptr = <double *> (data_ptr + value_offset)
        value[i] = value_ptr[0]

    return np.asarray(value)


cdef np.ndarray[np.int64_t, ndim=2] cget_int_values_2d(
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, byte_row, byte_col, nrow, ncol, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef int64_t *value_ptr

    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    byte_col = out.strides[1]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    ncol = out.shape[1]
    cdef int64_t [:, ::1] value = np.empty((nrow, ncol), dtype=np.int64) 
    for i in range(nrow):
        for j in range(ncol):
            data_ptr = base_ptr + i * byte_row + j * byte_col
            value_ptr = <int64_t *> (data_ptr + value_offset)
            value[i, j] = value_ptr[0]

    return np.asarray(value)


cdef np.ndarray[np.float64_t, ndim=2] cget_float_values_2d(
        str name,
        np.ndarray out,
):
  
    cdef Py_ssize_t i, j, byte_row, byte_col, nrow, ncol, value_offset

    # Declare pointers to the structured array 'out'
    cdef char *base_ptr = <char *> np.PyArray_DATA(out) 
    cdef char *data_ptr
    cdef double *value_ptr

    # Size for rows (in Bytes and Int)
    byte_row = out.strides[0]
    byte_col = out.strides[1]
    value_offset = <Py_ssize_t> out.dtype.fields[name][1]
    nrow = out.shape[0]
    ncol = out.shape[1]
    cdef double [:, ::1] value = np.empty((nrow, ncol), dtype=np.float64) 
    for i in range(nrow):
        for j in range(ncol):
            data_ptr = base_ptr + i * byte_row + j * byte_col
            value_ptr = <double *> (data_ptr + value_offset)
            value[i, j] = value_ptr[0]

    return np.asarray(value)


cpdef cget_values(
        str name,
        np.ndarray out,
):
    dtype = out.dtype.fields[name][0]
    if out.ndim == 0 and dtype == np.int64:
        value = cget_int_values_0d(name, out)
    elif out.ndim == 0 and dtype == np.float64:
        value = cget_float_values_0d(name, out)
    elif out.ndim == 1 and dtype == np.int64:
        value = cget_int_values_1d(name, out)
    elif out.ndim == 1 and dtype == np.float64:
        value = cget_float_values_1d(name, out)
    elif out.ndim == 2 and dtype == np.int64:
        value = cget_int_values_2d(name, out)
    elif out.ndim == 2 and dtype == np.float64:
        value = cget_float_values_2d(name, out)
    else:
        ValueError("Dimensions of 'out' argument must be 0, 1 or 2")

    return value
