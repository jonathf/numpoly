import numpy
import numpoly


def polynomial(
        poly_like=None,
        indeterminants="q",
        dtype=None,
):
    """
    Attempt to cast an object into a polynomial array.

    Supports various casting options:

    ``dict``
        Keys are tuples that represent polynomial exponents, and values are
        numpy arrays that represents polynomial coefficients.
    ``numpoly.ndpoly``
        Copy of the polynomial.
    ``int``, ``float``, ``numpy.ndarray``
        Constant term polynomial.
    ``sympy.Poly``
        Convert polynomial from ``sympy`` to ``numpoly``, if possible.
    ``Iterable``
        Multivariate array construction.
    ``numpy structured array``
        Assumes that the raw polynomial core. Used for developer
        convenience.

    Args:
        poly_like (Any):
            Input to be converted to a `numpoly.ndpoly` polynomial type.
        indeterminants (str, Tuple[str, ...]):
            Name of the indeterminant variables. If possible to infer from
            ``poly_like``, this argument will be ignored.
        dtype (type, numpy.dtype):
            Data type used for the polynomial coefficients.

    Returns:
        (numpoly.ndpoly):
            Polynomial based on input ``poly_like``.

    Examples:
        >>> print(numpoly.polynomial({(1,): 1}))
        q
        >>> x, y = numpoly.symbols("x y")
        >>> print(x**2 + x*y + 2)
        2+x*y+x**2
        >>> poly = -3*x + x**2 + y
        >>> print(numpoly.polynomial([x*y, x, y]))
        [x*y x y]
        >>> print(numpoly.polynomial([1, 2, 3]))
        [1 2 3]
        >>> import sympy
        >>> x_, y_ = sympy.symbols("x, y")
        >>> print(numpoly.polynomial(3*x_*y_ - 4 + x_**5))
        x**5+3*x*y-4
    """
    if poly_like is None:
        poly = numpoly.ndpoly(
            exponents=[(0,)],
            shape=(),
            indeterminants=indeterminants,
            dtype=dtype,
        )
        poly["0"] = 0

    elif isinstance(poly_like, dict):
        poly = numpoly.ndpoly(exponents=[(0,)], shape=())
        exponents, coefficients = zip(*list(poly_like.items()))
        poly = polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
            dtype=dtype,
        )

    elif isinstance(poly_like, numpoly.ndpoly):
        poly = poly_like.copy()

    # assume polynomial converted to structured array
    elif isinstance(poly_like, numpy.ndarray) and poly_like.dtype.names:
        exponents = numpoly.INVERSE_MAP(numpy.array([
            tuple(name) for name in poly_like.dtype.names], dtype="S"))
        coefficients = [poly_like[key] for key in poly_like.dtype.names]
        poly = polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
        )

    elif isinstance(poly_like, (int, float, numpy.ndarray, numpy.generic)):
        poly = polynomial_from_attributes(
            exponents=[(0,)],
            coefficients=numpy.array([poly_like]),
            indeterminants=indeterminants,
            dtype=dtype,
        )

    # handler for sympy objects
    elif hasattr(poly_like, "as_poly"):
        poly_like = poly_like.as_poly()
        exponents = poly_like.monoms()
        coefficients = [int(coeff) if coeff.is_integer else float(coeff)
                        for coeff in poly_like.coeffs()]
        indeterminants = [str(elem) for elem in poly_like.gens]
        poly = polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
        )

    else:
        poly = compose_polynomial_array(
            arrays=poly_like,
            dtype=dtype,
        )

    return poly


def polynomial_from_attributes(
        exponents,
        coefficients,
        indeterminants,
        dtype=None,
        trim=True,
):
    if trim:
        exponents, coefficients, indeterminants = clean_attributes(
            exponents, coefficients, indeterminants)
    dtype = coefficients[0].dtype if dtype is None else dtype
    poly = numpoly.ndpoly(
        exponents=exponents,
        shape=coefficients[0].shape,
        indeterminants=indeterminants,
        dtype=dtype,
    )
    for exponent, values in zip(poly._exponents, coefficients):
        poly[exponent] = values
    return poly


def clean_attributes(exponents, coefficients, indeterminants):
    coefficients = [numpy.asarray(coefficient)
                    for coefficient in coefficients]
    pairs = [
        (exponent, coefficient)
        for exponent, coefficient in zip(exponents, coefficients)
        if numpy.any(coefficient) or not any(exponent)
    ]
    if pairs:
        exponents, coefficients = zip(*pairs)
    else:
        exponents = [(0,)*len(indeterminants)]
        coefficients = numpy.zeros(
            (1,)+coefficients[0].shape, dtype=coefficients[0].dtype)

    exponents = numpy.asarray(exponents, dtype=int)
    if isinstance(indeterminants, numpoly.ndpoly):
        indeterminants = indeterminants._indeterminants

    if isinstance(indeterminants, str):
        if exponents.shape[1] > 1:
            indeterminants = ["%s%d" % (indeterminants, idx)
                              for idx in range(exponents.shape[1])]
        else:
            indeterminants = [indeterminants]
    assert len(indeterminants) == exponents.shape[1], (indeterminants, exponents)

    indices = numpy.any(exponents != 0, 0)
    assert exponents.shape[1] == len(indices), (exponents, indices)
    if not numpy.any(indices):
        indices[0] = True
    exponents = exponents[:, indices]
    assert exponents.size, (exponents, indices)
    indeterminants = numpy.array(indeterminants)[indices].tolist()

    assert len(exponents.shape) == 2, exponents
    assert len(exponents) == len(coefficients)
    assert len(numpy.unique(exponents, axis=0)) == exponents.shape[0], exponents
    assert sorted(set(indeterminants)) == sorted(indeterminants)

    return exponents, coefficients, indeterminants


def compose_polynomial_array(
        arrays,
        dtype=None,
):
    arrays = numpy.array(arrays, dtype=object)
    shape = arrays.shape
    arrays = arrays.flatten()

    indices = numpy.array([isinstance(array, numpoly.ndpoly)
                           for array in arrays])
    arrays[indices] = numpoly.align_polynomial_indeterminants(*arrays[indices])
    indeterminants = arrays[indices][0] if numpy.any(indices) else "q"
    arrays = arrays.tolist()

    dtypes = []
    keys = {(0,)}
    for array in arrays:
        if isinstance(array, numpoly.ndpoly):
            dtypes.append(array._dtype)
            keys = keys.union([tuple(key) for key in array.exponents.tolist()])
        elif isinstance(array, (numpy.generic, numpy.ndarray)):
            dtypes.append(array.dtype)
        else:
            dtypes.append(type(array))

    if dtype is None:
        dtype = numpy.find_common_type(dtypes, [])
    length = max([len(key) for key in keys])

    collection = {}
    for idx, array in enumerate(arrays):
        if isinstance(array, numpoly.ndpoly):
            for key, value in zip(array.exponents, array.coefficients):
                key = tuple(key)+(0,)*(length-len(key))
                if key not in collection:
                    collection[key] = numpy.zeros(len(arrays), dtype=dtype)
                collection[key][idx] = value
        else:
            key = (0,)*length
            if key not in collection:
                collection[key] = numpy.zeros(len(arrays), dtype=dtype)
            collection[key][idx] = array

    exponents = sorted(collection)
    coefficients = numpy.array([collection[key] for key in exponents])
    coefficients = coefficients.reshape(-1, *shape)

    exponents, coefficients, indeterminants = clean_attributes(
        exponents, coefficients, indeterminants)
    return polynomial_from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        indeterminants=indeterminants,
    )
