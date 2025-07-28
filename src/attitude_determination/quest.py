import numpy as np
from scipy.linalg import eigh

def adjoint_matrix(M):
    return np.linalg.det(M) * np.linalg.inv(M)

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
    def quest_algorithm(body_vectors, inertial_vectors, weights=None, tol=1e-12, max_iter=50):
        # Compute B, S, sigma, Z
        B, weights = AttitudeDetermination.compute_B(body_vectors, inertial_vectors, weights)
        S, sigma, Z = AttitudeDetermination.compute_S_sigma_Z(B)

        # λ0 fixed to 1
        lmbd = 1.0

        for _ in range(max_iter):
            f_val, f_prime, gamma, x = AttitudeDetermination.f_and_derivative(lmbd, S, sigma, Z)
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

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        q0, q1, q2, q3 = q
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