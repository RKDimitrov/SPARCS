import numpy as np
import pandas as pd
from scipy.linalg import eigh

# =========================
# CLASS TO LOAD HIPPARCOS CATALOG
# =========================
class HipparcosCatalog:
    def __init__(self, catalog_file):
        self.catalog = self._load_catalog(catalog_file)

    def _load_catalog(self, catalog_file):
        catalog_data = []
        try:
            with open(catalog_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'HIP' not in line or '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|') if p.strip()]
                try:
                    hip_id = int(parts[0].split()[1])
                    ra_h, ra_m, ra_s = [float(x) for x in parts[1].split()]
                    ra_deg = (ra_h + ra_m/60 + ra_s/3600) * 15.0
                    dec_parts = parts[2].split()
                    sign = -1 if dec_parts[0].startswith('-') else 1
                    dec_d = int(dec_parts[0].lstrip('+-'))
                    dec_m = int(dec_parts[1])
                    dec_s = float(dec_parts[2])
                    dec_deg = sign * (abs(dec_d) + dec_m/60 + dec_s/3600)
                    catalog_data.append({'HIP': hip_id, 'RA_deg': ra_deg, 'Dec_deg': dec_deg})
                except:
                    continue
            return pd.DataFrame(catalog_data)
        except Exception as e:
            print(f"Error loading catalog: {e}")
            return pd.DataFrame()

    def get_star_coords(self, hip_id):
        star = self.catalog[self.catalog["HIP"] == hip_id]
        if len(star) == 0:
            return None, None
        return star.iloc[0]["RA_deg"], star.iloc[0]["Dec_deg"]

# =========================
# QUEST ALGORITHM AND CONVERSIONS
# =========================
class AttitudeDetermination:

    @staticmethod
    def radec_to_unit_vector(ra_deg, dec_deg):
        ra = np.radians(ra_deg)
        dec = np.radians(dec_deg)
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.array([x, y, z])

    @staticmethod
    def quest_algorithm(body_vectors, inertial_vectors):
        n = len(body_vectors)
        weights = np.ones(n) / n
        B = np.zeros((3, 3))
        for i in range(n):
            B += weights[i] * np.outer(body_vectors[i], inertial_vectors[i])
        S = B + B.T
        sigma = np.trace(B)
        Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])
        K = np.zeros((4, 4))
        K[0,0] = sigma
        K[0,1:4] = Z
        K[1:4,0] = Z
        K[1:4,1:4] = S - sigma * np.eye(3)
        vals, vecs = eigh(K)
        q = vecs[:, np.argmax(vals)]
        if q[0] < 0: q = -q
        return q

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        q0,q1,q2,q3 = q
        return np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ])

    @staticmethod
    def rotation_matrix_to_euler(R):
        pitch = np.arcsin(-R[2, 0])
        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# =========================
# ATTITUDE CALCULATION
# =========================
def calculate_attitude(measurements_file, catalog_file):
    catalog = HipparcosCatalog(catalog_file)
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1,
                               names=["x","y","z","HIP_ID"])

    body_vectors, inertial_vectors, matched_stars = [], [], []

    for _, row in measurements.iterrows():
        ra, dec = catalog.get_star_coords(int(row["HIP_ID"]))
        if ra is None:
            continue
        body_vec = np.array([row["x"], row["y"], row["z"]])
        body_vec /= np.linalg.norm(body_vec)
        inertial_vec = AttitudeDetermination.radec_to_unit_vector(ra, dec)

        body_vectors.append(body_vec)
        inertial_vectors.append(inertial_vec)
        matched_stars.append(int(row["HIP_ID"]))

    if len(body_vectors) < 2:
        print("Error: at least 2 stars are required.")
        return None

    q = AttitudeDetermination.quest_algorithm(np.array(body_vectors), np.array(inertial_vectors))
    R = AttitudeDetermination.quaternion_to_rotation_matrix(q).T  # Transpose to get inertial → camera

    roll, pitch, yaw = AttitudeDetermination.rotation_matrix_to_euler(R)

    residuals = []
    for i in range(len(body_vectors)):
        rotated = R @ inertial_vectors[i]
        err = np.arccos(np.clip(np.dot(body_vectors[i], rotated), -1, 1))
        residuals.append(np.degrees(err))

    return {
        "quaternion": q,
        "rotation_matrix": R,
        "euler_angles": (roll, pitch, yaw),
        "matched_stars": matched_stars,
        "mean_error": np.mean(residuals),
        "max_error": np.max(residuals)
    }

# =========================
# PRINT RESULTS
# =========================
def print_results(results):
    if results is None: return
    q = results["quaternion"]
    roll,pitch,yaw = results["euler_angles"]
    R = results["rotation_matrix"]

    print("\n==============================")
    print("ATTITUDE DETERMINATION RESULTS")
    print("==============================")
    print(f"Quaternion: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    print(f"Roll  = {roll:.3f}°")
    print(f"Pitch = {pitch:.3f}°")
    print(f"Yaw   = {yaw:.3f}°")
    print("\nRotation Matrix:")
    for i in range(3):
        print(f"[{R[i,0]:8.5f} {R[i,1]:8.5f} {R[i,2]:8.5f}]")
    print(f"\nUsed {len(results['matched_stars'])} stars: {results['matched_stars']}")
    print(f"Mean error: {results['mean_error']:.4f}°")
    print(f"Max error:  {results['max_error']:.4f}°")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    catalog_file = r"C:\Users\157205\Desktop\SPARCS-main\HipparcosCatalog.txt"
    measurements_file = r"C:\Users\157205\Desktop\SPARCS-main\extracted_stars.txt"

    results = calculate_attitude(measurements_file, catalog_file)
    print_results(results)
