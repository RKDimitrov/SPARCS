import numpy as np
import pandas as pd
from .catalog import HipparcosCatalog
from .quest import AttitudeDetermination

# --- Helper functions ---
def _quat_mul(q2, q1):
    w2,x2,y2,z2 = q2
    w1,x1,y1,z1 = q1
    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ], dtype=float)

def _quat_normalize(q):
    n = np.linalg.norm(q)
    return q if n == 0 else q / n

def _quat_align_sign(q, ref):
    return -q if np.dot(q, ref) < 0 else q

def _quat_geodesic_deg(q1, q2):
    q1 = _quat_normalize(q1)
    q2 = _quat_align_sign(_quat_normalize(q2), q1)
    dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    return np.degrees(2.0 * np.arccos(dot))

def _reweight_cauchy(residuals_deg, c_deg=0.25):
    """
    Cauchy reweighting function with DEBUG info
    """
    r = np.asarray(residuals_deg, dtype=float)
    w_raw = 1.0 / (1.0 + (r / c_deg) ** 2)
    
    # DEBUG: Print weight analysis
    print(f"    🔍 IRLS Debug (c_deg = {c_deg:.6f}):")
    print(f"       Residuals: min={np.min(r):.3f}° max={np.max(r):.3f}° mean={np.mean(r):.3f}°")
    print(f"       Raw weights: min={np.min(w_raw):.6f} max={np.max(w_raw):.6f} mean={np.mean(w_raw):.6f}")
    
    # Count significantly downweighted stars using ABSOLUTE threshold
    low_weight_count_05 = np.sum(w_raw < 0.5)
    low_weight_count_01 = np.sum(w_raw < 0.1)
    low_weight_count_001 = np.sum(w_raw < 0.01)
    print(f"       Stars with raw weight < 0.5: {low_weight_count_05}/{len(r)}")
    print(f"       Stars with raw weight < 0.1: {low_weight_count_01}/{len(r)}")
    print(f"       Stars with raw weight < 0.01: {low_weight_count_001}/{len(r)}")
    
    # Normalize weights
    s = w_raw.sum()
    w_norm = (w_raw / s) if s > 0 else np.ones_like(w_raw) / len(w_raw)
    
    print(f"       Normalized weights: min={np.min(w_norm):.6f} max={np.max(w_norm):.6f}")
    
    # Show individual star weights for first few
    for i in range(min(5, len(r))):
        print(f"       Star {i}: error={r[i]:.3f}° → raw_weight={w_raw[i]:.6f} → norm_weight={w_norm[i]:.6f}")
    
    return w_norm, w_raw  # Return both normalized and raw weights

def extract_hip_int(hip_id):
    if isinstance(hip_id, int):
        return hip_id
    hip_id_str = str(hip_id).strip()
    if hip_id_str.startswith('HIP'):
        try:
            return int(hip_id_str.split()[1])
        except Exception:
            pass
    try:
        return int(hip_id_str)
    except Exception:
        raise ValueError(f"Could not extract HIP integer from: {hip_id}")

def count_kept_stars(w_raw, method='raw_threshold', threshold=0.5):
    """
    Different methods to count "kept" stars after IRLS
    
    Parameters:
    -----------
    w_raw : array
        Raw (unnormalized) weights from Cauchy function
    method : str
        'raw_threshold' - Use absolute threshold on raw weights
        'relative_mean' - Use relative threshold (old buggy method)
        'percentile' - Use percentile-based threshold
    threshold : float
        Threshold value (meaning depends on method)
    """
    if method == 'raw_threshold':
        # FIXED: Use absolute threshold on raw weights
        count = np.sum(w_raw >= threshold)
        print(f"       Count method: raw_threshold >= {threshold} → {count}/{len(w_raw)} stars kept")
        
    elif method == 'relative_mean':
        # OLD BUGGY METHOD: relative to mean
        mean_weight = np.mean(w_raw)
        relative_threshold = threshold * mean_weight
        count = np.sum(w_raw >= relative_threshold)
        print(f"       Count method: relative_mean >= {threshold}*{mean_weight:.6f} = {relative_threshold:.6f} → {count}/{len(w_raw)} stars kept")
        
    elif method == 'percentile':
        # Use percentile threshold
        percentile_threshold = np.percentile(w_raw, (1-threshold)*100)
        count = np.sum(w_raw >= percentile_threshold)
        print(f"       Count method: percentile >= {percentile_threshold:.6f} (top {threshold*100:.0f}%) → {count}/{len(w_raw)} stars kept")
        
    else:
        count = len(w_raw)
        print(f"       Count method: unknown '{method}' → keeping all {count} stars")
    
    return int(count)

# --- Main calculation ---
def calculate_attitude(measurements_file, catalog_file, max_iterations=50, c_deg=0.20, counting_method='raw_threshold', count_threshold=0.8):
    """
    FINAL VERSION with FIXED star counting logic
    
    Parameters:
    -----------
    c_deg : float
        Cauchy scale parameter for IRLS (default 0.25)
        - Smaller values = more strict outlier rejection
        - Larger values = more lenient
    counting_method : str
        Method to count "kept" stars: 'raw_threshold', 'relative_mean', 'percentile'
    count_threshold : float
        Threshold for counting (meaning depends on counting_method)
    """
    print(f"🔧 QUEST (FIXED COUNTING): Loading data from {measurements_file}")
    print(f"   IRLS settings: c_deg={c_deg}, counting='{counting_method}', threshold={count_threshold}")
    
    catalog = HipparcosCatalog(catalog_file)
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1, names=["x","y","z","HIP_ID"])
    body, inert, matched_stars = [], [], []

    print(f"📊 Processing {len(measurements)} measurements...")
    
    for i, row in measurements.iterrows():
        hip_int = extract_hip_int(row["HIP_ID"])
        ra, dec = catalog.get_star_coords(hip_int)
        if ra is None:
            print(f"  ⚠️  Star HIP {hip_int} not found in catalog")
            continue
        
        bvec = np.array([row["x"], row["y"], row["z"]])
        bvec_norm = np.linalg.norm(bvec)
        
        if bvec_norm == 0:
            print(f"  ⚠️  Zero vector for HIP {hip_int}")
            continue
            
        bvec /= bvec_norm
        rvec = AttitudeDetermination.radec_to_unit_vector(ra, dec)
        
        body.append(bvec)
        inert.append(rvec)
        matched_stars.append(hip_int)

    if len(body) < 2:
        print("❌ Error: at least 2 stars are required.")
        return None

    body = np.array(body)
    inert = np.array(inert)
    
    print(f"✅ Using {len(body)} matched stars for QUEST")

    # First QUEST iteration
    print("🎯 Running initial QUEST algorithm...")
    q_curr, lam_curr, iters_curr = AttitudeDetermination.quest_algorithm(body, inert, max_iter=max_iterations)
    R_curr = AttitudeDetermination.quaternion_to_rotation_matrix(q_curr).T

    # Initial residuals
    res = [np.degrees(np.arccos(np.clip(np.dot(b, R_curr @ r), -1.0, 1.0))) for b, r in zip(body, inert)]
    res = np.array(res, dtype=float)
    
    print(f"  Initial residuals: mean={np.mean(res):.3f}° max={np.max(res):.3f}°")

    stars_first_iter = len(res)

    # IRLS with FIXED counting
    MAX_IRLS_ITERS = 1
    TOL_Q_DEG = 1e-3

    stars_second_iter = stars_first_iter  # Default: keep all stars
    
    if MAX_IRLS_ITERS > 0:
        print(f"🔄 Running IRLS refinement...")
        
        for iteration in range(MAX_IRLS_ITERS):
            # Calculate weights with DEBUG info
            w_norm, w_raw = _reweight_cauchy(res, c_deg=c_deg)
            
            # FIXED: Count using the corrected method
            stars_second_iter = count_kept_stars(w_raw, method=counting_method, threshold=count_threshold)
            
            # Weighted QUEST
            q_next, lam_next, iters_next = AttitudeDetermination.quest_algorithm(body, inert, weights=w_norm, max_iter=max_iterations)
            q_next = _quat_align_sign(q_next, q_curr)
            R_next = AttitudeDetermination.quaternion_to_rotation_matrix(q_next).T

            # New residuals
            res_new = [np.degrees(np.arccos(np.clip(np.dot(b, R_next @ r), -1.0, 1.0))) for b, r in zip(body, inert)]
            res_new = np.array(res_new, dtype=float)
            
            print(f"  IRLS {iteration+1}: mean={np.mean(res_new):.3f}° max={np.max(res_new):.3f}°")
            
            if _quat_geodesic_deg(q_curr, q_next) < TOL_Q_DEG:
                print(f"  Converged after {iteration+1} IRLS iteration")
                q_curr, R_curr = q_next, R_next
                lam_curr, iters_curr = lam_next, iters_next
                res = res_new
                break

            q_curr, R_curr = q_next, R_next
            lam_curr, iters_curr = lam_next, iters_next
            res = res_new

    # Final results
    q_final = q_curr
    R_final = R_curr
    roll_final, pitch_final, yaw_final = AttitudeDetermination.rotation_matrix_to_euler(R_final)
    
    final_residuals = [np.degrees(np.arccos(np.clip(np.dot(b, R_final @ r), -1.0, 1.0))) for b, r in zip(body, inert)]
    
    mean_error = float(np.mean(final_residuals))
    max_error = float(np.max(final_residuals))
    
    print(f"🎯 QUEST FINAL Results: mean={mean_error:.3f}° max={max_error:.3f}°")

    return {
        "method": "QUEST (iterative reweighted)",
        "quaternion_camera": q_final,
        "quaternion_body": q_final,
        "rotation_matrix_body": R_final,
        "euler_angles_camera": (roll_final, pitch_final, yaw_final),
        "euler_angles_body": (roll_final, pitch_final, yaw_final),
        "matched_stars": matched_stars,
        "lambda_max": float(lam_curr),
        "mean_error": mean_error,
        "max_error": max_error,
        "Mean error": mean_error,
        "Max error": max_error,
        "newton_raphson_iterations": int(iters_curr),
        "stars_first_iter": int(stars_first_iter),
        "stars_second_iter": int(stars_second_iter),
        "c_deg_used": c_deg,
        "counting_method": counting_method,
        "count_threshold": count_threshold
    }

def print_results(results):
    if results is None: 
        print("❌ No results to print")
        return
        
    qc = results["quaternion_camera"]
    qb = results["quaternion_body"]
    roll_c, pitch_c, yaw_c = results["euler_angles_camera"]
    roll_b, pitch_b, yaw_b = results["euler_angles_body"]
    R = results["rotation_matrix_body"]

    print(f"\n{'='*60}")
    print(f"ATTITUDE DETERMINATION RESULTS ({results['method']})")
    print(f"{'='*60}")
    print(f"Quaternion (camera): [{qc[0]:.6f}, {qc[1]:.6f}, {qc[2]:.6f}, {qc[3]:.6f}]")
    print(f"Roll_c = {roll_c:.3f}°, Pitch_c = {pitch_c:.3f}°, Yaw_c = {yaw_c:.3f}°")
    print(f"\nQuaternion (body):   [{qb[0]:.6f}, {qb[1]:.6f}, {qb[2]:.6f}, {qb[3]:.6f}]")
    print(f"Roll_b = {roll_b:.3f}°, Pitch_b = {pitch_b:.3f}°, Yaw_b = {yaw_b:.3f}°")
    print("\nRotation Matrix (body):")
    for i in range(3):
        print(f"[{R[i,0]:8.5f} {R[i,1]:8.5f} {R[i,2]:8.5f}]")
    print(f"\nStars first iteration: {results['stars_first_iter']}")
    print(f"Stars second iteration: {results['stars_second_iter']}")
    print(f"IRLS parameters: c_deg={results.get('c_deg_used', 'N/A')}, method='{results.get('counting_method', 'N/A')}', threshold={results.get('count_threshold', 'N/A')}")
    print(f"\nUsed {len(results['matched_stars'])} stars total: {results['matched_stars']}")
    print(f"λmax: {results['lambda_max']:.8f}, Mean error: {results['mean_error']:.4f}°, Max error: {results['max_error']:.4f}°")
    
    if results['mean_error'] < 2:
        print("🏆 EXCELLENT attitude solution!")
    elif results['mean_error'] < 10:
        print("✅ GOOD attitude solution")
    elif results['mean_error'] < 45:
        print("⚠️  ACCEPTABLE but may need tuning")
    else:
        print("❌ POOR attitude solution - check coordinate systems")

