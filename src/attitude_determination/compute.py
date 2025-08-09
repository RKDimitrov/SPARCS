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
    r = np.asarray(residuals_deg, dtype=float)
    w = 1.0 / (1.0 + (r / c_deg) ** 2)
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / len(w)

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

# --- Main calculation ---
def calculate_attitude(measurements_file, catalog_file, max_iterations=50):
    catalog = HipparcosCatalog(catalog_file)
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1, names=["x","y","z","HIP_ID"])
    body, inert, matched_stars = [], [], []

    for _, row in measurements.iterrows():
        hip_int = extract_hip_int(row["HIP_ID"])
        ra, dec = catalog.get_star_coords(hip_int)
        if ra is None:
            continue
        bvec = np.array([row["x"], row["y"], row["z"]])
        bvec /= np.linalg.norm(bvec)
        rvec = AttitudeDetermination.radec_to_unit_vector(ra, dec)
        body.append(bvec)
        inert.append(rvec)
        matched_stars.append(hip_int)

    if len(body) < 2:
        print("Error: at least 2 stars are required.")
        return None

    body = np.array(body)
    inert = np.array(inert)

    # First QUEST
    q_curr, lam_curr, iters_curr = AttitudeDetermination.quest_algorithm(body, inert, max_iter=max_iterations)
    R_curr = AttitudeDetermination.quaternion_to_rotation_matrix(q_curr).T

    # Residuals
    res = [np.degrees(np.arccos(np.clip(np.dot(b, R_curr @ r), -1.0, 1.0))) for b, r in zip(body, inert)]
    res = np.array(res, dtype=float)

    # Count stars first iteration
    stars_first_iter = len(res)

    # IRLS
    IRLS_ITERS = 3
    TOL_Q_DEG  = 1e-3
    c_deg      = 0.25

    for _ in range(IRLS_ITERS):
        w = _reweight_cauchy(res, c_deg=c_deg)
        q_next, lam_next, iters_next = AttitudeDetermination.quest_algorithm(body, inert, weights=w, max_iter=max_iterations)
        q_next = _quat_align_sign(q_next, q_curr)
        R_next = AttitudeDetermination.quaternion_to_rotation_matrix(q_next).T

        res = [np.degrees(np.arccos(np.clip(np.dot(b, R_next @ r), -1.0, 1.0))) for b, r in zip(body, inert)]
        res = np.array(res, dtype=float)

        if _quat_geodesic_deg(q_curr, q_next) < TOL_Q_DEG:
            q_curr, R_curr = q_next, R_next
            lam_curr, iters_curr = lam_next, iters_next
            break

        q_curr, R_curr = q_next, R_next
        lam_curr, iters_curr = lam_next, iters_next

    # Count stars second iteration (weights > 0.5 are considered kept)
    stars_second_iter = np.sum(w > 0.5)

    # Save camera frame quaternion and Euler
    q_ic = q_curr
    roll_c, pitch_c, yaw_c = AttitudeDetermination.rotation_matrix_to_euler(R_curr)

    # Camera->Body
    q_cb = np.array([0.5, 0.5, -0.5, 0.5])
    q_ib = _quat_normalize(_quat_mul(q_ic, q_cb))
    R2 = AttitudeDetermination.quaternion_to_rotation_matrix(q_ib).T
    residuals2 = [np.degrees(np.arccos(np.clip(np.dot(b, R2 @ r), -1.0, 1.0))) for b, r in zip(body, inert)]
    roll_b, pitch_b, yaw_b = AttitudeDetermination.rotation_matrix_to_euler(R2)

    return {
        "method": "QUEST (iterative reweighted)",
        "quaternion_camera": q_ic,
        "quaternion_body": q_ib,
        "rotation_matrix_body": R2,
        "euler_angles_camera": (roll_c, pitch_c, yaw_c),
        "euler_angles_body": (roll_b, pitch_b, yaw_b),
        "matched_stars": matched_stars,
        "lambda_max": float(lam_curr),
        "mean_error": float(np.mean(residuals2)),
        "max_error": float(np.max(residuals2)),
        "newton_raphson_iterations": int(iters_curr),
        "stars_first_iter": int(stars_first_iter),
        "stars_second_iter": int(stars_second_iter)
    }

def print_results(results):
    if results is None: return
    qc = results["quaternion_camera"]
    qb = results["quaternion_body"]
    roll_c,pitch_c,yaw_c = results["euler_angles_camera"]
    roll_b,pitch_b,yaw_b = results["euler_angles_body"]
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
    print(f"\nUsed {len(results['matched_stars'])} stars total: {results['matched_stars']}")
    print(f"λmax: {results['lambda_max']:.8f}, Mean error: {results['mean_error']:.4f}°, Max error: {results['max_error']:.4f}°")
