import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
from collections import defaultdict

def pair_angle_matching_with_ids(catalog_vectors, catalog_ids, image_vectors, max_fov_deg=30, tolerance_deg=0.01):
    """
    Match stars from image to catalog using pairwise angular distances, returning HIP IDs.
    Args:
      catalog_vectors: np.ndarray, shape (N_cat, 3) - unit vectors for catalog stars
      catalog_ids: list or np.ndarray or pd.Series - catalog star IDs (e.g. HIP numbers)
      image_vectors: np.ndarray, shape (N_img, 3) - unit vectors for image stars
      max_fov_deg: float - max allowed angular separation between pairs (camera FoV)
      tolerance_deg: float - angular tolerance for matching pairs
    Returns:
      final_matches: list of dicts
        [{'image_star_index': i_img, 'catalog_star_index': i_cat, 'catalog_star_id': hip_id, 'votes': vote_count}, ...]
    """
    max_fov_rad = np.deg2rad(max_fov_deg)
    tolerance_rad = np.deg2rad(tolerance_deg)
    catalog_pairs = []
    catalog_pair_angles = []
    for i, j in combinations(range(len(catalog_vectors)), 2):
        dot = np.dot(catalog_vectors[i], catalog_vectors[j])
        dot = np.clip(dot, -1, 1)
        angle = np.arccos(dot)
        if angle <= max_fov_rad:
            catalog_pairs.append((i, j))
            catalog_pair_angles.append([angle])
    catalog_pair_angles = np.array(catalog_pair_angles)
    catalog_tree = KDTree(catalog_pair_angles)
    image_pairs = []
    image_pair_angles = []
    for i, j in combinations(range(len(image_vectors)), 2):
        dot = np.dot(image_vectors[i], image_vectors[j])
        dot = np.clip(dot, -1, 1)
        angle = np.arccos(dot)
        if angle <= max_fov_rad:
            image_pairs.append((i, j))
            image_pair_angles.append([angle])
    image_pair_angles = np.array(image_pair_angles)
    match_votes = defaultdict(int)
    for idx, angle_feat in enumerate(image_pair_angles):
        nearby_indices = catalog_tree.query_ball_point(angle_feat, r=tolerance_rad)
        i_img, j_img = image_pairs[idx]
        for cat_idx in nearby_indices:
            i_cat, j_cat = catalog_pairs[cat_idx]
            match_votes[(i_img, i_cat)] += 1
            match_votes[(j_img, j_cat)] += 1
    vote_table = defaultdict(lambda: defaultdict(int))
    for (img_idx, cat_idx), votes in match_votes.items():
        vote_table[img_idx][cat_idx] += votes
    final_matches = []
    for img_idx, cat_votes in vote_table.items():
        best_cat_idx = max(cat_votes, key=cat_votes.get)
        final_matches.append({
            'image_star_index': img_idx,
            'catalog_star_index': best_cat_idx,
            'catalog_star_id': catalog_ids[best_cat_idx],
            'votes': cat_votes[best_cat_idx]
        })
    return final_matches 