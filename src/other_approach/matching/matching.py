import numpy as np

def match_stars(image_vecs, catalog_df, max_angle_deg=5.0):
    max_angle_rad = np.deg2rad(max_angle_deg)
    matched_catalog_vecs = []
    matched_indices = []
    catalog_vecs = catalog_df[['x', 'y', 'z']].values
    for vec in image_vecs:
        dots = catalog_vecs @ vec
        angles = np.arccos(np.clip(dots, -1, 1))
        min_idx = np.argmin(angles)
        if angles[min_idx] < max_angle_rad:
            matched_catalog_vecs.append(catalog_vecs[min_idx])
            matched_indices.append(min_idx)
    return np.array(matched_catalog_vecs), matched_indices 