import numpy as np
import pandas as pd

def trace_adj_3x3(S: np.ndarray) -> float:
    trS = np.trace(S)
    trS2 = np.trace(S @ S)
    return 0.5 * (trS * trS - trS2)

def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])

def quaternion_to_rotation_matrix_IB(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),         1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),         2*(q2*q3 + q0*q1),         1 - 2*(q1**2 + q2**2)]
    ])

def rotation_matrix_to_euler_ZYX_BI(R_BI: np.ndarray):
    pitch = np.arcsin(-R_BI[2, 0])
    cp = np.cos(pitch)
    if cp > 1e-6:
        roll = np.arctan2(R_BI[2, 1], R_BI[2, 2])
        yaw  = np.arctan2(R_BI[1, 0], R_BI[0, 0])
    else:
        roll = 0.0
        yaw  = np.arctan2(-R_BI[0, 1], R_BI[1, 1])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

class HipparcosCatalog:
    def __init__(self, catalog_file: str):
        self.catalog = self._load_catalog(catalog_file)

    def _load_catalog(self, catalog_file: str) -> pd.DataFrame:
        rows = []
        try:
            with open(catalog_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'HIP' not in line or '|' not in line:
                        continue
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    try:
                        hip_str = parts[0].split()
                        if hip_str[0] != 'HIP':
                            continue
                        hip_id = int(hip_str[1])
                        ra_h, ra_m, ra_s = [float(x) for x in parts[1].split()]
                        ra_deg = (ra_h + ra_m/60.0 + ra_s/3600.0) * 15.0
                        dec_parts = parts[2].split()
                        sign = -1 if dec_parts[0].startswith('-') else 1
                        dec_d = int(dec_parts[0].lstrip('+-'))
                        dec_m = int(dec_parts[1])
                        dec_s = float(dec_parts[2])
                        dec_deg = sign * (abs(dec_d) + dec_m/60.0 + dec_s/3600.0)
                        rows.append({'HIP': hip_id, 'RA_deg': ra_deg, 'Dec_deg': dec_deg})
                    except Exception:
                        continue
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error loading catalog: {e}")
            return pd.DataFrame()

    def get_star_coords(self, hip_id: int):
        star = self.catalog[self.catalog["HIP"] == hip_id]
        return (None, None) if len(star) == 0 else (float(star.iloc[0]["RA_deg"]), float(star.iloc[0]["Dec_deg"]))

class QUEST:
    @staticmethod
    def compute_B(body_vectors: np.ndarray,
                  inertial_vectors: np.ndarray,
                  weights: np.ndarray | None = None):
        n = len(body_vectors)
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / np.sum(weights)
        B = np.zeros((3, 3))
        for b, r, w in zip(body_vectors, inertial_vectors, weights):
            B += w * np.outer(b, r)
        return B, weights

    @staticmethod
    def compute_S_sigma_Z(B: np.ndarray):
        S = B + B.T
        sigma = np.trace(B)
        Z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ])
        return S, sigma, Z

    @staticmethod
    def f_and_derivative(lmbd: float, S: np.ndarray, sigma: float, Z: np.ndarray):
        tr_adjS = trace_adj_3x3(S)
        alpha = lmbd**2 - sigma**2 + tr_adjS
        M = alpha * np.eye(3) + (lmbd - sigma) * S + S @ S
        x = M @ Z
        A = (lmbd + sigma) * np.eye(3) - S
        gamma = np.linalg.det(A)
        f_val = gamma * (lmbd - sigma) - Z.T @ x
        delta = 1e-6 * max(1.0, abs(lmbd))
        l2 = lmbd + delta
        alpha2 = l2**2 - sigma**2 + tr_adjS
        M2 = alpha2 * np.eye(3) + (l2 - sigma) * S + S @ S
        x2 = M2 @ Z
        A2 = (l2 + sigma) * np.eye(3) - S
        gamma2 = np.linalg.det(A2)
        f_val2 = gamma2 * (l2 - sigma) - Z.T @ x2
        f_prime = (f_val2 - f_val) / delta
        return f_val, f_prime, gamma, x

    @staticmethod
    def quest_method(body_vectors: np.ndarray,
                     inertial_vectors: np.ndarray,
                     weights: np.ndarray | None = None,
                     tol: float = 1e-12,
                     max_iter: int = 50):
        B, weights = QUEST.compute_B(body_vectors, inertial_vectors, weights)
        S, sigma, Z = QUEST.compute_S_sigma_Z(B)
        lmbd = float(np.sum(weights))
        iters = 0
        for iters in range(1, max_iter + 1):
            f_val, f_prime, gamma, x = QUEST.f_and_derivative(lmbd, S, sigma, Z)
            if abs(f_prime) < 1e-14:
                break
            step = -f_val / f_prime
            step = float(np.clip(step, -1.0, 1.0))
            lmbd += step
            if abs(step) < tol:
                break
        norm_factor = np.sqrt(gamma**2 + np.dot(x, x))
        q = np.array([gamma, x[0], x[1], x[2]]) / norm_factor
        if q[0] < 0:
            q = -q
        return q, lmbd, iters

def rotation_matrix_to_euler_ZYX_BI(R_BI: np.ndarray):
    pitch = np.arcsin(-R_BI[2, 0])
    cp = np.cos(pitch)
    if cp > 1e-6:
        roll = np.arctan2(R_BI[2, 1], R_BI[2, 2])
        yaw  = np.arctan2(R_BI[1, 0], R_BI[0, 0])
    else:
        roll = 0.0
        yaw  = np.arctan2(-R_BI[0, 1], R_BI[1, 1])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def quat_mul(q2, q1):
    w2,x2,y2,z2 = q2
    w1,x1,y1,z1 = q1
    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ], dtype=float)

def calculate_attitude_quest(measurements_file: str,
                             catalog_file: str,
                             max_iterations: int = 50):
    catalog = HipparcosCatalog(catalog_file)
    measurements = pd.read_csv(measurements_file, sep="\t", skiprows=1, names=["x", "y", "z", "HIP_ID"])
    body_vectors, inertial_vectors, matched_stars = [], [], []
    for _, row in measurements.iterrows():
        try:
            hip_id = int(row["HIP_ID"])
        except Exception:
            continue
        ra, dec = catalog.get_star_coords(hip_id)
        if ra is None:
            continue
        b = np.array([row["x"], row["y"], row["z"]], dtype=float)
        norm_b = np.linalg.norm(b)
        if norm_b <= 0:
            continue
        b /= norm_b
        r = radec_to_unit_vector(ra, dec)
        body_vectors.append(b)
        inertial_vectors.append(r)
        matched_stars.append(hip_id)
    if len(body_vectors) < 2:
        return None
    body_vectors = np.array(body_vectors)
    inertial_vectors = np.array(inertial_vectors)

    q, lambda_max, iters = QUEST.quest_method(body_vectors, inertial_vectors, max_iter=max_iterations)
    C_IB = quaternion_to_rotation_matrix_IB(q)
    R_BI = C_IB.T
    roll, pitch, yaw = rotation_matrix_to_euler_ZYX_BI(R_BI)

    residuals = []
    for b, r in zip(body_vectors, inertial_vectors):
        br = R_BI @ r
        cosang = np.clip(np.dot(b, br), -1.0, 1.0)
        residuals.append(np.degrees(np.arccos(cosang)))

    c_deg = 0.25
    resid_arr = np.array(residuals, dtype=float)
    w = 1.0 / (1.0 + (resid_arr / c_deg) ** 2)
    w /= w.sum()

    q2, lambda_max2, iters2 = QUEST.quest_method(body_vectors, inertial_vectors,
                                                 weights=w, max_iter=max_iterations)
    C_IB2 = quaternion_to_rotation_matrix_IB(q2)
    R_BI2 = C_IB2.T

    residuals = []
    for b, r in zip(body_vectors, inertial_vectors):
        br = R_BI2 @ r
        cosang = np.clip(np.dot(b, br), -1.0, 1.0)
        residuals.append(np.degrees(np.arccos(cosang)))

    q_ic = q2
    q_cb = np.array([0.5, 0.5, -0.5, 0.5])  # coniugato di q_bc
    q_ib = quat_mul(q_ic, q_cb)

    q_ib = q_ib / np.linalg.norm(q_ib)
    C_IB_out = quaternion_to_rotation_matrix_IB(q_ib)
    R_BI_out = C_IB_out.T
    roll_out, pitch_out, yaw_out = rotation_matrix_to_euler_ZYX_BI(R_BI_out)

    return {
        "method": "QUEST (one-shot reweighted)",
        "quaternion": q_ib,
        "rotation_matrix": R_BI_out,
        "euler_angles": (roll_out, pitch_out, yaw_out),
        "matched_stars": matched_stars,
        "lambda_max": lambda_max2,
        "mean_error": float(np.mean(residuals)),
        "max_error": float(np.max(residuals)),
        "newton_raphson_iterations": iters2
    }
#ciao
def print_results(results: dict | None):
    if results is None:
        return
    q = results["quaternion"]
    roll, pitch, yaw = results["euler_angles"]
    R = results["rotation_matrix"]
    print(f"\n{'='*60}")
    print(f"ATTITUDE DETERMINATION RESULTS ({results['method']})")
    print(f"{'='*60}")
    print(f"Quaternion (inertial -> body): [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    print(f"Roll = {roll:.3f}°, Pitch = {pitch:.3f}°, Yaw = {yaw:.3f}°")
    for i in range(3):
        print(f"[{R[i,0]:8.5f} {R[i,1]:8.5f} {R[i,2]:8.5f}]")
    print(f"\nUsed {len(results['matched_stars'])} stars: {results['matched_stars']}")
    print(f"λmax: {results['lambda_max']:.8f}, "
          f"Mean error: {results['mean_error']:.4f}°, "
          f"Max error: {results['max_error']:.4f}°, "
          f"Newton iters: {results['newton_raphson_iterations']}")

if __name__ == "__main__":
    catalog_file = r"C:\Users\157205\Desktop\SPARCS-main\HipparcosCatalog.txt"
    measurements_file = r"C:\Users\157205\Desktop\SPARCS-main\extracted_stars.txt"
    results = calculate_attitude_quest(measurements_file, catalog_file, max_iterations=50)
    print_results(results)
