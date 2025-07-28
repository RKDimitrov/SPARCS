import numpy as np
import pandas as pd

def adjoint_matrix(M):
    return np.linalg.det(M) * np.linalg.inv(M)

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
                    dec_m, dec_s = int(dec_parts[1]), float(dec_parts[2])
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
        return (None, None) if len(star) == 0 else (star.iloc[0]["RA_deg"], star.iloc[0]["Dec_deg"])

class QUEST:
    @staticmethod
    def compute_B(body_vectors, inertial_vectors, weights=None):
        n = len(body_vectors)
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = weights / np.sum(weights)
        B = np.zeros((3, 3))
        for b, r, w in zip(body_vectors, inertial_vectors, weights):
            B += w * np.outer(b, r)
        return B, weights

    @staticmethod
    def compute_S_sigma_Z(B):
        S = B + B.T
        sigma = np.trace(B)
        Z = np.array([B[1, 2] - B[2, 1],
                      B[2, 0] - B[0, 2],
                      B[0, 1] - B[1, 0]])
        return S, sigma, Z

    @staticmethod
    def f_and_derivative(lmbd, S, sigma, Z):
        adjS = adjoint_matrix(S)
        alpha = lmbd**2 - sigma**2 + np.trace(adjS)
        M = alpha * np.eye(3) + (lmbd - sigma) * S + S @ S
        x = M @ Z
        A = (lmbd + sigma) * np.eye(3) - S
        gamma = np.linalg.det(A)
        f_val = gamma * (lmbd - sigma) - Z.T @ x
        delta = 1e-8
        l2 = lmbd + delta
        alpha2 = l2**2 - sigma**2 + np.trace(adjS)
        M2 = alpha2 * np.eye(3) + (l2 - sigma) * S + S @ S
        x2 = M2 @ Z
        A2 = (l2 + sigma) * np.eye(3) - S
        gamma2 = np.linalg.det(A2)
        f_val2 = gamma2 * (l2 - sigma) - Z.T @ x2
        f_prime = (f_val2 - f_val) / delta
        return f_val, f_prime, gamma, x

    @staticmethod
    def quest_method(body_vectors, inertial_vectors, weights=None, tol=1e-12, max_iter=50):
        # Calcolo di B, S, sigma, Z
        B, weights = QUEST.compute_B(body_vectors, inertial_vectors, weights)
        S, sigma, Z = QUEST.compute_S_sigma_Z(B)

        # λ0 fissato a 1
        lmbd = 1.0

        for _ in range(max_iter):
            f_val, f_prime, gamma, x = QUEST.f_and_derivative(lmbd, S, sigma, Z)
            if abs(f_prime) < 1e-14:
                break
            delta = -f_val / f_prime
            lmbd += delta
            if abs(delta) < tol:
                break

        norm_factor = np.sqrt(gamma**2 + np.dot(x, x))
        q = np.array([gamma, x[0], x[1], x[2]]) / norm_factor
        if q[0] < 0:
            q = -q
        return q, lmbd

def radec_to_unit_vector(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])

def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])

def rotation_matrix_to_euler(R):
    pitch = np.arcsin(-R[2, 0])
    if np.cos(pitch) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def calculate_attitude_quest(measurements_file, catalog_file, max_iterations=50):
    catalog = HipparcosCatalog(catalog_file)
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1, names=["x", "y", "z", "HIP_ID"])
    body_vectors, inertial_vectors, matched_stars = [], [], []
    for _, row in measurements.iterrows():
        ra, dec = catalog.get_star_coords(int(row["HIP_ID"]))
        if ra is None:
            continue
        body_vec = np.array([row["x"], row["y"], row["z"]])
        body_vec /= np.linalg.norm(body_vec)
        inertial_vec = radec_to_unit_vector(ra, dec)
        body_vectors.append(body_vec)
        inertial_vectors.append(inertial_vec)
        matched_stars.append(int(row["HIP_ID"]))
    if len(body_vectors) < 2:
        print("Error: at least 2 stars are required.")
        return None
    q, lambda_max = QUEST.quest_method(np.array(body_vectors), np.array(inertial_vectors), max_iter=max_iterations)
    R = quaternion_to_rotation_matrix(q).T
    roll, pitch, yaw = rotation_matrix_to_euler(R)
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
    if results is None:
        return
    q, (roll, pitch, yaw), R = results["quaternion"], results["euler_angles"], results["rotation_matrix"]
    print(f"\n{'='*60}\nATTITUDE DETERMINATION RESULTS ({results['method']})\n{'='*60}")
    print(f"Quaternion: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    print(f"Roll = {roll:.3f}°, Pitch = {pitch:.3f}°, Yaw = {yaw:.3f}°")
    print("\nRotation Matrix:")
    for i in range(3):
        print(f"[{R[i,0]:8.5f} {R[i,1]:8.5f} {R[i,2]:8.5f}]")
    print(f"\nUsed {len(results['matched_stars'])} stars: {results['matched_stars']}")
    print(f"λmax: {results['lambda_max']:.8f}, Mean error: {results['mean_error']:.4f}°, Max error: {results['max_error']:.4f}°")

if __name__ == "__main__":
    catalog_file = r"C:\Users\157205\Desktop\SPARCS-main\HipparcosCatalog.txt"
    measurements_file = r"C:\Users\157205\Desktop\SPARCS-main\extracted_stars.txt"
    results = calculate_attitude_quest(measurements_file, catalog_file, max_iterations=50)
    print_results(results)
