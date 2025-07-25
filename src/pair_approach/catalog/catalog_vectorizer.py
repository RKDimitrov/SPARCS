import numpy as np

def get_catalog_vectors_and_ids(hip, id_col='name'):
    """Return catalog vectors and IDs from DataFrame."""
    vectors = hip[['x', 'y', 'z']].values
    ids = hip[id_col].values
    return vectors, ids 