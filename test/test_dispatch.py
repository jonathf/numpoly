from numpoly import dispatch


def test_collection_sizes():
    collection_diff = set(dispatch.FUNCTION_COLLECTION).symmetric_difference(dispatch.UFUNC_COLLECTION)

