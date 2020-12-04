"""Align polynomials."""
import numpy
import numpoly


def align_polynomials(*polys):
    """
    Align polynomial such that dimensionality, shape, etc. are compatible.

    Alignment includes shape (for broadcasting), indeterminants, exponents and
    dtype.

    Args:
        polys (numpoly.ndpoly):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly.ndpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.

    Examples:
        >>> q0 = numpoly.variable()
        >>> q0q1 = numpoly.variable(2)
        >>> q0
        polynomial(q0)
        >>> q0.coefficients
        [1]
        >>> q0.indeterminants
        polynomial([q0])
        >>> q0, _ = numpoly.align_polynomials(q0, q0q1.astype(float))
        >>> q0
        polynomial([q0, q0])
        >>> q0.coefficients
        [array([1., 1.]), array([0., 0.])]
        >>> q0.indeterminants
        polynomial([q0, q1])

    """
    # polys = numpoly.broadcast_arrays(*polys)
    polys = align_shape(*polys)
    polys = align_indeterminants(*polys)
    polys = align_exponents(*polys)
    polys = align_dtype(*polys)
    return polys


def align_shape(*polys):
    """
    Align polynomial by shape.

    Args:
        polys (numpoly.ndpoly):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly.ndpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly1 = 4*q0
        >>> poly2 = numpoly.polynomial([[2*q0+1, 3*q0-q1]])
        >>> poly1.shape
        ()
        >>> poly2.shape
        (1, 2)
        >>> poly1, poly2 = numpoly.align_shape(poly1, poly2)
        >>> poly1
        polynomial([[4*q0, 4*q0]])
        >>> poly2
        polynomial([[2*q0+1, -q1+3*q0]])
        >>> poly1.shape
        (1, 2)
        >>> poly2.shape
        (1, 2)

    """
    # return tuple(numpoly.broadcast_arrays(*polys))
    polys = [numpoly.aspolynomial(poly) for poly in polys]
    common = 1
    for poly in polys:
        if poly.size:
            common = numpy.ones(poly.coefficients[0].shape, dtype=int)*common

    polys = [poly.from_attributes(
        exponents=poly.exponents,
        coefficients=[coeff*common for coeff in poly.coefficients],
        names=poly.indeterminants,
    ) for poly in polys]
    assert numpy.all(common.shape == poly.shape for poly in polys)
    return tuple(polys)


def align_indeterminants(*polys):
    """
    Align polynomial by indeterminants.

    Args:
        polys (numpoly.ndpoly):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly.ndpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.

    Examples:
        >>> q0 = numpoly.variable()
        >>> poly1 = numpoly.polynomial(2*q0+1)
        >>> q0, q1 = numpoly.variable(2)
        >>> poly2 = numpoly.polynomial(3*q0-q1)
        >>> poly1.indeterminants
        polynomial([q0])
        >>> poly2.indeterminants
        polynomial([q0, q1])
        >>> poly1, poly2 = numpoly.align_indeterminants(poly1, poly2)
        >>> poly1
        polynomial(2*q0+1)
        >>> poly2
        polynomial(-q1+3*q0)
        >>> poly1.indeterminants
        polynomial([q0, q1])
        >>> poly2.indeterminants
        polynomial([q0, q1])

    """
    polys = [numpoly.aspolynomial(poly) for poly in polys]
    common_names = sorted({name for poly in polys for name in poly.names})
    if not common_names:
        return polys

    for idx, poly in enumerate(polys):
        indices = numpy.array([
            common_names.index(name)
            for name in poly.names
            if name in common_names
        ])
        exponents = numpy.zeros(
            (len(poly.keys), len(common_names)), dtype=int)
        if indices.size:
            exponents[:, indices] = poly.exponents
        polys[idx] = poly.from_attributes(
            exponents=exponents,
            coefficients=poly.coefficients,
            names=common_names,
            retain_coefficients=True,
            retain_names=True,
        )
    assert all([polys[0].names == poly.names for poly in polys])

    return tuple(polys)


def align_exponents(*polys):
    """
    Align polynomials such that the exponents are the same.

    Aligning exponents assumes that the indeterminants is also aligned.

    Args:
        polys (numpoly.ndpoly):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly.ndpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly1 = q0*q1
        >>> poly2 = numpoly.polynomial([q0**5, q1**3-1])
        >>> poly1.exponents
        array([[1, 1]], dtype=uint32)
        >>> poly2.exponents
        array([[0, 0],
               [0, 3],
               [5, 0]], dtype=uint32)
        >>> poly1, poly2 = numpoly.align_exponents(poly1, poly2)
        >>> poly1
        polynomial(q0*q1)
        >>> poly2
        polynomial([q0**5, q1**3-1])
        >>> poly1.exponents
        array([[1, 1],
               [0, 0],
               [0, 3],
               [5, 0]], dtype=uint32)
        >>> poly2.exponents
        array([[1, 1],
               [0, 0],
               [0, 3],
               [5, 0]], dtype=uint32)

    """
    polys = [numpoly.aspolynomial(poly) for poly in polys]
    if not all(
            polys[0].names == poly.names
            for poly in polys
    ):
        polys = list(align_indeterminants(*polys))

    global_exponents = [tuple(exponent) for exponent in polys[0].exponents]

    for poly in polys[1:]:
        global_exponents.extend([tuple(exponent)
                                 for exponent in poly.exponents
                                 if tuple(exponent) not in global_exponents])

    for idx, poly in enumerate(polys):
        lookup = {
            tuple(exponent): coefficient
            for exponent, coefficient in zip(
                poly.exponents, poly.coefficients)
        }

        zeros = numpy.zeros(poly.shape, dtype=poly.dtype)
        coefficients = [lookup.get(exponent, zeros)
                        for exponent in global_exponents]
        polys[idx] = poly.from_attributes(
            exponents=global_exponents,
            coefficients=coefficients,
            names=poly.names,
            retain_coefficients=True,
            retain_names=True,
        )
    return tuple(polys)


def align_dtype(*polys):
    """
    Align polynomial by shape.

    Args:
        polys (numpoly.ndpoly):
            Polynomial to make adjustment to.

    Returns:
        (Tuple[numpoly.ndpoly, ...]):
            Same as ``polys``, but internal adjustments made to make them
            compatible for further operations.

    Examples:
        >>> q0 = numpoly.variable()
        >>> q0.dtype.name
        'int64'
        >>> poly, _ = numpoly.align_dtype(q0, 4.5)
        >>> poly.dtype.name
        'float64'

    """
    polys = [numpoly.aspolynomial(poly) for poly in polys]
    dtype = numpy.sum([numpy.array(True, dtype=poly.dtype) for poly in polys]).dtype
    polys = [numpoly.ndpoly.from_attributes(
        exponents=poly.exponents,
        coefficients=poly.coefficients,
        names=poly.indeterminants,
        dtype=dtype,
        retain_coefficients=True,
        retain_names=True,
    ) for poly in polys]
    return tuple(polys)
