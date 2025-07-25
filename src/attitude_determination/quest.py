import numpy as np
from scipy.linalg import eigh

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