import numpy as np

def solve_wahba(cam_vecs, inertial_vecs):
    B = cam_vecs.T @ inertial_vecs
    U, _, Vt = np.linalg.svd(B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R 