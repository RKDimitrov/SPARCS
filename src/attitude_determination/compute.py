import numpy as np
import pandas as pd
import os
import sys
from .catalog import HipparcosCatalog
from .quest import AttitudeDetermination

# Add path for database imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pair_approach'))
from database.db_utils import DatabaseManager

def extract_hip_int(hip_id):
    """Extract integer HIP number from a string like 'HIP 73273' or just an int."""
    if isinstance(hip_id, int):
        return hip_id
    if isinstance(hip_id, float):
        return int(hip_id)
    hip_id_str = str(hip_id).strip()
    if hip_id_str.startswith('HIP'):
        try:
            return int(hip_id_str.split()[1])
        except Exception:
            pass
    # fallback: try to convert directly
    try:
        return int(float(hip_id_str))  # Handle both string and float cases
    except Exception:
        raise ValueError(f"Could not extract HIP integer from: {hip_id}")

def calculate_attitude_from_database(measurements_file, max_iterations=50):
    """
    Calculate attitude using database instead of CSV file.
    This is the optimized version that doesn't need to load HipparcosCatalog.txt.
    """
    # Load database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'pair_approach', 'database', 'star_catalog.db')
    db = DatabaseManager(db_path)
    
    # Read measurements
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1, names=["x","y","z","HIP_ID"])
    body_vectors, inertial_vectors, matched_stars = [], [], []
    
    for _, row in measurements.iterrows():
        hip_int = extract_hip_int(row["HIP_ID"])
        
        # Get star coordinates from database
        star_data = db.get_star_by_hip(hip_int)
        if star_data is None:
            continue
            
        ra_deg = star_data['ra_deg']
        dec_deg = star_data['dec_deg']
        
        body_vec = np.array([row["x"], row["y"], row["z"]])
        body_vec /= np.linalg.norm(body_vec)
        inertial_vec = AttitudeDetermination.radec_to_unit_vector(ra_deg, dec_deg)
        body_vectors.append(body_vec)
        inertial_vectors.append(inertial_vec)
        matched_stars.append(hip_int)
    
    if len(body_vectors) < 2:
        print("Error: at least 2 stars are required.")
        return None
    
    q, lambda_max = AttitudeDetermination.quest_algorithm(np.array(body_vectors), np.array(inertial_vectors), max_iter=max_iterations)
    R = AttitudeDetermination.quaternion_to_rotation_matrix(q).T
    roll, pitch, yaw = AttitudeDetermination.rotation_matrix_to_euler(R)
    residuals = [np.degrees(np.arccos(np.clip(np.dot(body_vectors[i], R @ inertial_vectors[i]), -1, 1)))
                 for i in range(len(body_vectors))]
    
    return {
        "method": "QUEST (Database)",
        "quaternion": q,
        "rotation_matrix": R,
        "euler_angles": (roll, pitch, yaw),
        "matched_stars": matched_stars,
        "lambda_max": lambda_max,
        "mean_error": np.mean(residuals),
        "max_error": np.max(residuals),
        "newton_raphson_iterations": max_iterations
    }

def calculate_attitude(measurements_file, catalog_file, max_iterations=50):
    """
    Original function for backward compatibility - uses CSV file.
    """
    catalog = HipparcosCatalog(catalog_file)
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1, names=["x","y","z","HIP_ID"])
    body_vectors, inertial_vectors, matched_stars = [], [], []
    for _, row in measurements.iterrows():
        hip_int = extract_hip_int(row["HIP_ID"])
        ra, dec = catalog.get_star_coords(hip_int)
        if ra is None:
            continue
        body_vec = np.array([row["x"], row["y"], row["z"]])
        body_vec /= np.linalg.norm(body_vec)
        inertial_vec = AttitudeDetermination.radec_to_unit_vector(ra, dec)
        body_vectors.append(body_vec)
        inertial_vectors.append(inertial_vec)
        matched_stars.append(hip_int)
    if len(body_vectors) < 2:
        print("Error: at least 2 stars are required.")
        return None
    q, lambda_max = AttitudeDetermination.quest_algorithm(np.array(body_vectors), np.array(inertial_vectors), max_iter=max_iterations)
    R = AttitudeDetermination.quaternion_to_rotation_matrix(q).T
    roll, pitch, yaw = AttitudeDetermination.rotation_matrix_to_euler(R)
    residuals = [np.degrees(np.arccos(np.clip(np.dot(body_vectors[i], R @ inertial_vectors[i]), -1, 1)))
                 for i in range(len(body_vectors))]
    return {
        "method": "QUEST",
        "quaternion": q,
        "rotation_matrix": R,
        "euler_angles": (roll, pitch, yaw),
        "matched_stars": matched_stars,
        "lambda_max": lambda_max,
        "mean_error": np.mean(residuals),
        "max_error": np.max(residuals),
        "newton_raphson_iterations": max_iterations
    }

def print_results(results):
    if results is None: return
    q = results["quaternion"]
    roll,pitch,yaw = results["euler_angles"]
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
    print(f"λmax: {results['lambda_max']:.8f}, Mean error: {results['mean_error']:.4f}°, Max error: {results['max_error']:.4f}°") 