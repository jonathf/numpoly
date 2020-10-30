"""Save a polynomial array to a text file."""
import numpy
from numpy.lib.recfunctions import structured_to_unstructured
import numpoly

from ..dispatch import implements


HEADER_TEMPLATE = "numpoly:{version} names:{names} keys:{keys} shape:{shape}"


@implements(numpy.savetxt)
def savetxt(fname, X, fmt="%.18e", delimiter=" ", newline="\n",
            header="", footer="", comments="# ", encoding=None):
    """
    Save a polynomial array to a text file.

    To store a polynomial to text string, the polynomial is converted through
    a flat structured array, to a matrix. Extra meta information about the
    indeterminant names, exponent keys and array shape are all stored
    separate at the top of the header of the output file.

    Args:
        fname (str, pathlib.Path, filehandle):
            If the filename ends in ``.gz``, the file is automatically saved in
            compressed gzip format. `loadtxt` understands gzipped files
            transparently.
        X (numpoly.ndpoly):
            Data to be saved to a text file.
        fmt (str, Sequence[str]):
            A single format (%10.5f), a sequence of formats, or a multi-format
            string, e.g. 'Iteration %d -- %10.5f', in which case `delimiter` is
            ignored. For complex `X`, the legal options for `fmt` are:

            * A single specifier, `fmt='%.4e'`, resulting in numbers formatted
              like `' (%s+%sj)' % (fmt, fmt)`.
            * A full string specifying every real and imaginary part, e.g.
              `' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'` for 3 columns.
            * A list of specifiers, one per column - in this case, the real
              and imaginary part must have separate specifiers,
              e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns.

        delimiter (str):
            String or character separating columns.
        newline (str):
            String or character separating lines.
        header (str):
            String that will be written at the beginning of the file.
        footer (str):
            String that will be written at the end of the file.
        comments (str):
            String that will be prepended to the ``header`` and ``footer``
            strings, to mark them as comments. Default: '# ',  as expected by
            e.g. ``numpoly.loadtxt``.
        encoding (Optional[str]):
            Encoding used to encode the outputfile. Does not apply to output
            streams. If the encoding is something other than 'bytes' or
            'latin1' you will not be able to load the file in NumPy versions <
            1.14. Default is 'latin1'.

    Examples:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([1, q0, q1**2-1])
        >>> numpoly.savetxt("/tmp/poly.txt", poly)
        >>> numpoly.loadtxt("/tmp/poly.txt")
        polynomial([1.0, q0, q1**2-1.0])
        >>> numpoly.savetxt("/tmp/poly.txt", poly, header="my header")
        >>> numpoly.loadtxt("/tmp/poly.txt", skiprows=1)
        polynomial([1.0, q0, q1**2-1.0])

    """
    if isinstance(X, numpoly.ndpoly):
        numpoly_header = HEADER_TEMPLATE.format(
            version=numpoly.__version__,
            names=",".join(X.names),
            keys=",".join(X.keys),
            shape=",".join(str(idx) for idx in X.shape)
        )
        if header:
            header = numpoly_header + "\n" + header
        else:
            header = numpoly_header
        X = structured_to_unstructured(X.values.ravel())

    numpy.savetxt(fname=fname, X=X, fmt=fmt, delimiter=delimiter,
                  newline=newline, header=header, footer=footer,
                  comments=comments, encoding=encoding)
