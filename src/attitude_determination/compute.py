import numpy as np
import pandas as pd
from .catalog import HipparcosCatalog
from .quest import AttitudeDetermination

def extract_hip_int(hip_id):
    """Extract integer HIP number from a string like 'HIP 73273' or just an int."""
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

def calculate_attitude(measurements_file, catalog_file, max_iterations=50):
    catalog = HipparcosCatalog(catalog_file)
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1, names=["x","y","z","HIP_ID"])

    body_vectors, inertial_vectors, matched_stars = [], [], []
    for _, row in measurements.iterrows():
        hip_int = extract_hip_int(row["HIP_ID"])
        ra, dec = catalog.get_star_coords(hip_int)
        if ra is None:
            continue
        b = np.array([row["x"], row["y"], row["z"]], dtype=float)
        nb = np.linalg.norm(b)
        if nb <= 0:
            continue
        b /= nb
        r = AttitudeDetermination.radec_to_unit_vector(ra, dec)
        body_vectors.append(b)
        inertial_vectors.append(r)
        matched_stars.append(hip_int)

    if len(body_vectors) < 2:
        print("Error: at least 2 stars are required.")
        return None

    body = np.array(body_vectors)
    inert = np.array(inertial_vectors)

    # 1) QUEST base
    q, lambda_max, iters = AttitudeDetermination.quest_algorithm(body, inert, max_iter=max_iterations)
    R = AttitudeDetermination.quaternion_to_rotation_matrix(q).T

    # 2) residui e reweight (Cauchy-like), poi secondo run
    residuals = [np.degrees(np.arccos(np.clip(np.dot(b, R @ r), -1.0, 1.0)))
                 for b, r in zip(body, inert)]
    c_deg = 0.25  # tarabile: 0.10–0.50 in base al rumore
    w = 1.0 / (1.0 + (np.asarray(residuals) / c_deg) ** 2)
    w /= w.sum()

    q2, lambda_max2, iters2 = AttitudeDetermination.quest_algorithm(body, inert, weights=w, max_iter=max_iterations)
    R2 = AttitudeDetermination.quaternion_to_rotation_matrix(q2).T
    residuals2 = [np.degrees(np.arccos(np.clip(np.dot(b, R2 @ r), -1.0, 1.0)))
                  for b, r in zip(body, inert)]
    roll, pitch, yaw = AttitudeDetermination.rotation_matrix_to_euler(R2)

    return {
        "method": "QUEST (one-shot reweighted)",
        "quaternion": q2,
        "rotation_matrix": R2,
        "euler_angles": (roll, pitch, yaw),
        "matched_stars": matched_stars,
        "lambda_max": float(lambda_max2),
        "mean_error": float(np.mean(residuals2)),
        "max_error": float(np.max(residuals2)),
        "newton_raphson_iterations": int(iters2)
    }

def print_results(results):
    if results is None: 
        return
    q = results["quaternion"]
    roll, pitch, yaw = results["euler_angles"]
    R = results["rotation_matrix"]
    print(f"\n{'='*60}")
    print(f"ATTITUDE DETERMINATION RESULTS ({results['method']})")
    print(f"{'='*60}")
    print(f"Quaternion: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    print(f"Roll = {roll:.3f}°, Pitch = {pitch:.3f}°, Yaw = {yaw:.3f}°")
    print("\nRotation Matrix:")
    for i in range(3):
        print(f"[{R[i,0]:8.5f} {R[i,1]:8.5f} {R[i,2]:8.5f}]")
    print(f"\nUsed {len(results['matched_stars'])} stars: {results['matched_stars']}")
    print(f"λmax: {results['lambda_max']:.8f}, "
          f"Mean error: {results['mean_error']:.4f}°, "
          f"Max error: {results['max_error']:.4f}°")
