"""Implementation wrapper."""
ARRAY_FUNCTIONS = {}


def implements(*numpy_functions):
    """Register an __array_function__ implementation for Polynomial objects."""
    def decorator(numpoly_function):
        """Register function."""
        for numpy_function in numpy_functions:
            ARRAY_FUNCTIONS[numpy_function] = numpoly_function
        return numpoly_function

    return decorator
