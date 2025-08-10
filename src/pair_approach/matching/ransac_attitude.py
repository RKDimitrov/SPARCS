import numpy as np
import random

# ---------- Utilities ----------
def unit_rows(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def angle_between(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    cross = np.linalg.norm(np.cross(u, v))
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.rad2deg(np.arctan2(cross, dot))

def procrustes_rotation(img_vecs, cat_vecs):
    """
    Stima R (image->catalog) via SVD/Orthogonal Procrustes.
    """
    img = np.asarray(img_vecs, dtype=float)
    cat = np.asarray(cat_vecs, dtype=float)
    
    # Normalizza
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    cat /= np.linalg.norm(cat, axis=1, keepdims=True)

    # SVD per Procrustes
    B = cat.T @ img
    U, s, Vt = np.linalg.svd(B)
    Rmat = U @ Vt
    
    # Assicura det(R) = +1
    if np.linalg.det(Rmat) < 0:
        U[:, -1] *= -1
        Rmat = U @ Vt
    
    return Rmat

def _unique_one_to_one_with_angle(matches, image_vectors, catalog_vectors):
    """Greedy 1-a-1 ordinando per angolo crescente, poi triad_votes, poi votes."""
    if not matches:
        return []
    
    def key(m):
        iv = image_vectors[m['image_star_index']]
        cv = catalog_vectors[m['catalog_star_index']]
        ang = angle_between(iv, cv)
        return (ang, -m.get('triad_votes', 0), -m.get('votes', 0))
    
    ms = sorted(matches, key=key)
    used_img, used_cat, out = set(), set(), []
    for m in ms:
        i = m['image_star_index']; c = m['catalog_star_index']
        if i in used_img or c in used_cat:
            continue
        used_img.add(i); used_cat.add(c)
        out.append(m)
    return out

# ---------- RANSAC ----------
def solve_rotation_from_seeds(seeds, image_vectors, catalog_vectors):
    """
    seeds: lista di dict con 'image_star_index', 'catalog_star_index'
    Ritorna R (3x3) oppure None se trio degenerato/duplicato.
    """
    img_vecs, cat_vecs = [], []
    used_img, used_cat = set(), set()
    
    for m in seeds:
        i = int(m["image_star_index"])
        c = int(m["catalog_star_index"])
        if i in used_img or c in used_cat:
            continue
        img_vecs.append(image_vectors[i])
        cat_vecs.append(catalog_vectors[c])
        used_img.add(i); used_cat.add(c)

    if len(img_vecs) < 3:
        return None

    try:
        # Verifica che i punti non siano collineari
        img_arr = np.array(img_vecs)
        cat_arr = np.array(cat_vecs)
        
        # Check per degenerazione
        if len(img_arr) >= 3:
            # Calcola il volume del tetraedro formato dai primi 3 punti
            v1, v2, v3 = img_arr[:3]
            vol = abs(np.dot(v1, np.cross(v2, v3)))
            if vol < 1e-6:  # Troppo piccolo = collineare
                return None
        
        return procrustes_rotation(img_arr, cat_arr)
    except (np.linalg.LinAlgError, ValueError):
        return None

def ransac_refine_matches(image_vectors, catalog_vectors, catalog_ids,
                          seed_matches, angle_thresh_deg=2.0,  # AUMENTATA da 1.0
                          n_iterations=500, min_inliers=4,  # PIÙ iterazioni
                          return_rotation=False):
    """
    RANSAC sulla rotazione con parametri più permissivi:
      - campiona 3 seed, stima R
      - proietta tutte le stelle immagine
      - nearest neighbor nel catalogo
      - conta inlier entro angle_thresh_deg
    Deduplica 1-a-1 prima del return.
    """
    print(f"Starting RANSAC with {len(seed_matches)} seed matches")
    print(f"Parameters: angle_thresh={angle_thresh_deg}°, iterations={n_iterations}")
    
    if len(seed_matches) < 3:
        print("Not enough seed matches for RANSAC")
        return [] if not return_rotation else ([], np.eye(3))

    image_vectors  = unit_rows(image_vectors)
    catalog_vectors = unit_rows(catalog_vectors)

    best_inliers, best_R = [], None
    best_score = 0

    # Prepara mapping per debug
    seed_map = {(m['image_star_index'], m['catalog_star_index']): m for m in seed_matches}

    successful_iterations = 0
    failed_rotations = 0

    for iteration in range(n_iterations):
        # Campiona 3 seed diversi
        trio = random.sample(seed_matches, 3)
        Rmat = solve_rotation_from_seeds(trio, image_vectors, catalog_vectors)
        
        if Rmat is None:
            failed_rotations += 1
            continue
        
        successful_iterations += 1

        # Proietta tutte le stelle immagine nel frame catalogo
        rv_all = (Rmat @ image_vectors.T).T      # (Nimg,3) nel frame catalogo
        dots   = catalog_vectors @ rv_all.T      # (Ncat, Nimg)
        idxs   = np.argmax(dots, axis=0)         # nearest neighbor per ogni stella immagine

        # Valuta gli inlier
        cand = []
        inlier_angles = []
        
        for i_img, i_cat in enumerate(idxs):
            ang = angle_between(rv_all[i_img], catalog_vectors[i_cat])
            if ang <= angle_thresh_deg:
                cand.append({
                    'image_star_index': i_img,
                    'catalog_star_index': int(i_cat),
                    'catalog_star_id':  catalog_ids[int(i_cat)],
                    'angle_err_deg':    float(ang)
                })
                inlier_angles.append(ang)

        # Score basato su numero + qualità degli inlier
        if cand:
            avg_error = np.mean(inlier_angles)
            score = len(cand) - 0.1 * avg_error  # Favorisce più inlier con errori minori
        else:
            score = 0

        if score > best_score:
            best_inliers = cand
            best_R = Rmat
            best_score = score
            print(f"  Iteration {iteration}: New best with {len(cand)} inliers (avg error: {avg_error:.3f}°)")

    print(f"RANSAC completed: {successful_iterations} successful / {failed_rotations} failed rotations")
    print(f"Best solution: {len(best_inliers)} inliers")

    # Refine finale usando tutti gli inlier trovati
    if best_inliers:
        # Deduplica 1-a-1 per evitare HIP ripetuti
        print("Applying 1-to-1 matching constraint...")
        best_inliers = _unique_one_to_one_with_angle(best_inliers, image_vectors, catalog_vectors)
        print(f"After deduplication: {len(best_inliers)} unique matches")

        # Re-stima la rotazione con tutti gli inlier unici
        if len(best_inliers) >= 3:
            img_vecs = np.array([image_vectors[m['image_star_index']] for m in best_inliers])
            cat_vecs = np.array([catalog_vectors[m['catalog_star_index']] for m in best_inliers])
            try:
                refined_R = procrustes_rotation(img_vecs, cat_vecs)
                
                # Verifica la qualità del refinement
                refined_rv = (refined_R @ img_vecs.T).T
                final_errors = [angle_between(refined_rv[i], cat_vecs[i]) for i in range(len(img_vecs))]
                avg_final_error = np.mean(final_errors)
                max_final_error = np.max(final_errors)
                
                print(f"Final rotation quality: avg error {avg_final_error:.3f}°, max error {max_final_error:.3f}°")
                
                # Aggiorna gli errori nei match
                for i, m in enumerate(best_inliers):
                    m['angle_err_deg'] = final_errors[i]
                
                best_R = refined_R
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Failed to refine rotation: {e}")

    else:
        print("No inliers found")

    if return_rotation:
        return best_inliers, (best_R if best_R is not None else np.eye(3))
    return best_inliers

def evaluate_rotation_quality(R, image_vectors, catalog_vectors, matches):
    """
    Valuta la qualità di una rotazione sui match forniti.
    """
    if not matches or R is None:
        return {}
    
    img_indices = [m['image_star_index'] for m in matches]
    cat_indices = [m['catalog_star_index'] for m in matches]
    
    img_vecs = image_vectors[img_indices]
    cat_vecs = catalog_vectors[cat_indices]
    
    # Proietta le stelle immagine
    projected = (R @ img_vecs.T).T
    
    # Calcola errori
    errors = [angle_between(projected[i], cat_vecs[i]) for i in range(len(projected))]
    
    return {
        'mean_error_deg': np.mean(errors),
        'max_error_deg': np.max(errors),
        'min_error_deg': np.min(errors),
        'std_error_deg': np.std(errors),
        'num_matches': len(matches)
    }