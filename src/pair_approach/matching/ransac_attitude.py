import numpy as np
import random
from attitude_determination.compute import calculate_attitude
from scipy.spatial.transform import Rotation as R

def ransac_refine_matches(image_vectors, catalog_vectors, catalog_ids,
                          seed_matches, angle_thresh_deg=0.5,
                          n_iterations=50):
    """
    Use seed_matches to solve for rotation, then project *all* image_vectors
    and assign nearest catalog star within angle_thresh_deg.
    """
    best_inliers = []

    for _ in range(n_iterations):
        # 1) Randomly pick 3 seeds
        trio = random.sample(seed_matches, 3)
        # 2) build a temporary QUEST input file to get rotation matrix
        #    or call calculate_attitude directly if available as function
        Rmat = solve_rotation_from_seeds(trio, image_vectors, catalog_vectors)

        inliers = []
        for i, iv in enumerate(image_vectors):
            rv = Rmat.dot(iv)
            # find closest catalog star
            dots = catalog_vectors.dot(rv)
            idx = np.argmax(dots)
            ang = np.rad2deg(np.arccos(np.clip(dots[idx], -1, 1)))
            if ang <= angle_thresh_deg:
                inliers.append({
                    'image_star_index': i,
                    'catalog_star_index': idx,
                    'catalog_star_id': catalog_ids[idx]
                })
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    return best_inliers

def calculate_attitude_from_seeds(seeds, image_vectors):
    """
    Compute the rotation matrix from image to catalog frame
    using at least 3 matching vector pairs.
    Args:
        seeds: list of (image_vec, catalog_vec) tuples
        image_vectors: array of all image vectors
    Returns:
        R: 3x3 rotation matrix (np.ndarray)
        success: bool
    """
    import numpy as np

    B = np.zeros((3, 3))
    for img_vec, cat_vec in seeds:
        B += np.outer(cat_vec, img_vec)

    # SVD of B
    U, _, Vt = np.linalg.svd(B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, True


def solve_rotation_from_seeds(seeds, image_vectors, catalog_vectors):
    #print("[DEBUG] Seeds:", seeds)
    seed_vectors = []
    img_vectors, cat_vectors = [], []
    for match in seeds:
        img_idx = match["image_star_index"]
        cat_idx = match["catalog_star_index"]
        img_vec = image_vectors[img_idx]
        cat_vec = catalog_vectors[cat_idx]
        img_vectors.append(img_vec)
        cat_vectors.append(cat_vec)


    Rmat, success = calculate_attitude_from_seeds(seed_vectors, image_vectors)
    return Rmat
