import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
from collections import defaultdict
from scipy.spatial import KDTree
from collections import Counter

def pair_angle_matching_with_ids(catalog_vectors, catalog_ids, image_vectors,
                                 intensities, catalog_vmag,
                                 max_fov_deg=66, tolerance_deg=0.05):
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
        
        
    # --- Vote‐based pre‐filter: keep only the top K most‐voted matches ---
    K = min(len(final_matches), 15)      # keep up to 15 strongest matches
    final_matches = sorted(final_matches,
                           key=lambda m: m['votes'],
                           reverse=True)[:K]

    # If you want *at least* 3 seeds for triads/RANSAC, ensure K≥3
    if len(final_matches) < 3:
        print(f"[Warning] Only {len(final_matches)} matches after vote‐filter. "
              "Reverting to top 3 matches for stability.")
        final_matches = sorted(final_matches,
                               key=lambda m: m['votes'],
                               reverse=True)[:3]

    return final_matches




def filter_by_direct_angle(matches, image_vectors, catalog_vectors,
                           max_angle_deg):
    """
    Remove any matches whose image–catalog angular separation 
    exceeds max_angle_deg.
    """
    filtered = []
    for m in matches:
        iv = image_vectors[m['image_star_index']]
        cv = catalog_vectors[m['catalog_star_index']]
        ang = np.rad2deg(np.arccos(np.clip(np.dot(iv, cv), -1, 1)))
        if ang <= max_angle_deg:
            filtered.append(m)
    return filtered



def triad_refinement(image_vectors, catalog_vectors, initial_matches,
                     side_tol_deg=0.02, angle_tol_deg=0.5, vote_threshold=2):
    """
    Refine initial pair‐based matches with triangle (triad) consistency.
    Args:
      image_vectors: (N_img,3) array.
      catalog_vectors: (N_cat,3) array.
      initial_matches: list of dicts from pair_angle_matching_with_ids.
      side_tol_deg: tolerance for side‐length matching (deg).
      angle_tol_deg: tolerance for the included angle (deg).
      vote_threshold: min votes (triangles) to keep a match.
    Returns:
      refined_matches: same format as initial_matches but pruned/exacted.
    """
    # Build reverse map: image_idx → catalog_idx
    img2cat = {m['image_star_index']: m['catalog_star_index']
               for m in initial_matches}

    image_idxs = list(img2cat.keys())
    catalog_idxs = [img2cat[i] for i in image_idxs]

    # Precompute triangle features for catalog on the fly:
    # We'll need a KDTree on (side1, side2, angle).
    tri_feats = []
    tri_map = []  # parallel list of (i,j,k) catalog idxs
    for (a, b, c) in combinations(catalog_idxs, 3):
        va, vb, vc = catalog_vectors[a], catalog_vectors[b], catalog_vectors[c]
        # compute side-angles
        def ang(u,v):
            d = np.clip(np.dot(u,v), -1,1)
            return np.rad2deg(np.arccos(d))
        lab = ang(va, vb)
        lac = ang(va, vc)
        # angle at a via spherical law of cosines on triangle sides:
        # here, approximate with planar law since FOV small:
        # angle = arccos((lab^2 + lac^2 - lbc^2)/(2*lab*lac))
        lbc = ang(vb, vc)
        cosA = np.clip((lab**2 + lac**2 - lbc**2) / (2*lab*lac), -1,1)
        A = np.rad2deg(np.arccos(cosA))
        tri_feats.append([lab, lac, A])
        tri_map.append((a, b, c))

    catalog_tree = KDTree(tri_feats)

    # Now vote: for each image triangle
    vote_counts = Counter()
    for (i, j, k) in combinations(image_idxs, 3):
        vi, vj, vk = image_vectors[i], image_vectors[j], image_vectors[k]
        # same feature extraction
        def ang_img(u,v):
            d = np.clip(np.dot(u,v), -1,1)
            return np.rad2deg(np.arccos(d))
        lij = ang_img(vi, vj)
        lik = ang_img(vi, vk)
        ljk = ang_img(vj, vk)
        cosI = np.clip((lij**2 + lik**2 - ljk**2) / (2*lij*lik), -1,1)
        I = np.rad2deg(np.arccos(cosI))
        # query catalog triangles within tolerance
        query_feat = [lij, lik, I]
        radius = np.sqrt(side_tol_deg**2 + side_tol_deg**2 + angle_tol_deg**2)
        idxs = catalog_tree.query_ball_point(query_feat, r=radius)

        # each matched triangle votes for its three point correspondences
        for tri_idx in idxs:
            ca, cb, cc = tri_map[tri_idx]
            vote_counts[(i, ca)] += 1
            vote_counts[(j, cb)] += 1
            vote_counts[(k, cc)] += 1

    # Build refined list: keep only those with enough votes
    refined = []
    for m in initial_matches:
        key = (m['image_star_index'], m['catalog_star_index'])
        if vote_counts[key] >= vote_threshold:
            # bump up the vote count to the triad votes
            refined.append({
                **m,
                'triad_votes': vote_counts[key]
            })
    return refined
