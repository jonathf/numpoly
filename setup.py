from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension module
extensions = [
    Extension(
        name="numpoly.cfunctions.cmultiply",  # Name of the extension module
        sources=["numpoly/cfunctions/cmultiply.pyx"],  # Path to your `.pyx` file
        include_dirs=[np.get_include()],  # Include NumPy headers
    )
]

# Setup configuration
setup(
    ext_modules=cythonize(extensions),  # Compile the Cython files
    include_dirs=[np.get_include()],  # Include NumPy headers
)
