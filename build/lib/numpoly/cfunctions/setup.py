import os
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# setup.py to compile the Cython code
for file in os.listdir():
    if ".pyx" in file:
        setup(
            ext_modules=cythonize(file),  # Your .pyx file
            include_dirs=[np.get_include()]  # Include NumPy headers
        )
