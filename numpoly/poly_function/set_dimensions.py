"""Adjust the dimensions of a polynomial."""
import numpy
import numpoly


def set_dimensions(poly, dimensions=None):
    """
    Adjust the dimensions of a polynomial.

    Args:
        poly (numpoly.ndpoly):
            Input polynomial
        dimensions (int):
            The dimensions of the output polynomial. If omitted, increase
            polynomial with one dimension. If the new dim is smaller then
            `poly`'s dimensions, variables with cut components are all cut.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = q0*q1-q0**2
        >>> numpoly.set_dimensions(poly, 1)
        polynomial(-q0**2)
        >>> numpoly.set_dimensions(poly, 3)
        polynomial(q0*q1-q0**2)
        >>> numpoly.set_dimensions(poly).names
        ('q0', 'q1', 'q2')

    """
    poly = numpoly.aspolynomial(poly)
    if dimensions is None:
        dimensions = len(poly.names)+1
    diff = dimensions-len(poly.names)
    if diff > 0:
        padding = numpy.zeros((len(poly.exponents), diff), dtype="uint32")
        exponents = numpy.hstack([poly.exponents, padding])
        coefficients = poly.coefficients
        varname = numpoly.get_options()["default_varname"]
        names = list(poly.names)
        idx = 0
        while len(names) < dimensions:
            candidate = "%s%d" % (varname, idx)
            if candidate not in names:
                names.append(candidate)
            idx += 1

        indices = numpy.lexsort([names])
        exponents = exponents[:, indices]
        names = [names[idx] for idx in indices]


    elif diff < 0:
        indices = True ^ numpy.any(poly.exponents[:, dimensions:], -1)
        exponents = poly.exponents[:, :dimensions]
        exponents = exponents[indices]
        coefficients = [
            coeff for coeff, idx in zip(poly.coefficients, indices) if idx]
        names = poly.names[:dimensions]

    else:
        return poly

    return numpoly.polynomial_from_attributes(
        exponents=exponents,
        coefficients=coefficients,
        names=names,
        dtype=poly.dtype,
        allocation=poly.allocation,
        retain_names=True,
    )
