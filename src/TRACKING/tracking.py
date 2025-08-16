##TRACKING CODE TESTING USING JAPANESE METHOD
#Quaternion: [0.370812, 0.114763, 0.047753, 0.920352]
#Quaternion: [0.386422, -0.035837, -0.308044, 0.868621]

# N approach_2.py
# Tracking-mode single step:
# - hard-codes q1 from Lost-in-Space
# - predicts/star-matches on image 2
# - re-estimates attitude with your QUEST
# - prints q2_est (scalar-last [qx,qy,qz,qw], inertial->body)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set
import numpy as np
import math
import importlib.util, sys
from collections import deque

# =============================================================================
# 0) Bring in your existing LiS helpers (star detect, intrinsics, QUEST, etc.)
# =============================================================================
MODULE_PATH = r"C:\Users\mubas\OneDrive\Documents\Modules\Year 1\Endurosat\SPARCS\src\NEW ALTERNATE APPROACH\N approach_2.py"
MODULE_NAME = "lis_utils"

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
lis = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = lis
spec.loader.exec_module(lis)

# LiS utilities we’ll reuse
QUEST = lis.QUEST
detect_stars = lis.detect_stars
parse_catalog = lis.parse_catalog
intrinsics_from_fov = lis.intrinsics_from_fov
pixel_to_ray_stereo_f = lis.pixel_to_ray_stereo_f
ray_to_pixel_stereo_f = lis.ray_to_pixel_stereo_f
quaternion_angular_distance = lis.quaternion_angular_distance

# =============================================================================
# 1) Types & Config
# =============================================================================
Vec3 = np.ndarray
QuatSL = np.ndarray  # [qx,qy,qz,qw] scalar-last, inertial->body

@dataclass
class CameraIntrinsics:
    f_pix: float
    cx: float
    cy: float
    width: int
    height: int

@dataclass
class TrackConfig:
    neighbors_per_star: int = 9        # paper Sec. 3.2 (≤9)
    tracking_radius_px: float = 5.0    # paper Sec. 3.3 (d=5 px)
    min_vectors_for_quest: int = 3
    mag_limit: float = 4.5
    # If your LiS quats are B->I (scalar-last), set this to False to auto-conjugate them.
    quats_are_I2B: bool = True
    debug: bool = True

# =============================================================================
# 2) Quaternion utilities (scalar-last, I->B)
# =============================================================================
def q_norm(q: QuatSL) -> QuatSL:
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return q if n == 0 else q / n

def q_conj(q: QuatSL) -> QuatSL:
    return np.array([-q[0], -q[1], -q[2], q[3]], float)

def q_mul(q2: QuatSL, q1: QuatSL) -> QuatSL:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    v = np.array([
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ], float)
    w = w2*w1 - (x2*x1 + y2*y1 + z2*z1)
    return q_norm(np.array([*v, w], float))

def q_to_dcm_BI(q: QuatSL) -> np.ndarray:
    x, y, z, w = q_norm(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)]
    ], float)

def quat_log(q: QuatSL) -> Vec3:
    q = q_norm(q)
    v = q[:3]; w = q[3]
    nv = np.linalg.norm(v)
    if nv < 1e-15:
        return np.zeros(3)
    theta = 2.0 * math.atan2(nv, w)     # total angle
    return (theta / nv) * v             # axis*angle

def quat_exp(v: Vec3) -> QuatSL:
    th = float(np.linalg.norm(v))
    if th < 1e-15:
        return np.array([0.0,0.0,0.0,1.0], float)
    u = v / th
    s = math.sin(th/2.0)
    return q_norm(np.array([u[0]*s, u[1]*s, u[2]*s, math.cos(th/2.0)], float))

def angular_velocity_from_two_quats(q_nm1: QuatSL, q_n: QuatSL, dt: float) -> Vec3:
    # q_rel = q_n ⊗ q_{n-1}^* ; ω ≈ (2/dt) * log(q_rel)
    q_rel = q_mul(q_n, q_conj(q_nm1))
    v = quat_log(q_rel)
    return (2.0/dt) * v     # rad/s in body

def predict_next_quat(q_n: QuatSL, omega_body: Vec3, dt: float) -> QuatSL:
    # q_{n+1|est} = exp( (dt/2) * ω ) ⊗ q_n
    q_delta = quat_exp(0.5*dt * omega_body)
    return q_mul(q_delta, q_n)

# =============================================================================
# 3) Neighbor catalog (≤9 nearest per star)
# =============================================================================
def build_neighbor_catalog(catalog: List[dict], k: int = 9) -> Dict[int, List[int]]:
    R = np.array([s["vec"] for s in catalog], float)  # inertial unit vectors
    neighbors: Dict[int, List[int]] = {}
    for i in range(len(catalog)):
        r0 = R[i]
        dots = np.clip(R @ r0, -1.0, 1.0)
        order = np.argsort(-dots)                     # largest dot = smallest angle
        neigh = [int(j) for j in order if j != i][:k]
        neighbors[i] = neigh
    return neighbors

# =============================================================================
# 4) Imaging model (I -> B -> C -> pixel)
# =============================================================================
def inertial_to_pixel(r_i: Vec3, C_BI: np.ndarray, C_CB: np.ndarray, K: CameraIntrinsics) -> Optional[Tuple[float,float]]:
    v_body = C_BI @ r_i
    v_body /= (np.linalg.norm(v_body) + 1e-15)
    v_cam  = C_CB @ v_body
    x, y, ok = ray_to_pixel_stereo_f(v_cam, K.cx, K.cy, K.f_pix)
    if not ok or x < 0 or x >= K.width or y < 0 or y >= K.height:
        return None
    return (float(x), float(y))

def pixel_to_body_ray(u: float, v: float, K: CameraIntrinsics, C_CB: np.ndarray) -> Vec3:
    v_cam = pixel_to_ray_stereo_f(u, v, K.cx, K.cy, K.f_pix)
    v_body = C_CB.T @ v_cam
    return v_body / (np.linalg.norm(v_body) + 1e-15)

# =============================================================================
# 5) Predict stars that stay in FOV; robust seeding/fallback
# =============================================================================
def predict_star_pixels_next(
    q_nm1: QuatSL,
    q_n: QuatSL,
    dt: float,
    known_ids_at_n: Iterable[str],   # star IDs identified at step n (LiS output)
    catalog: List[dict],
    neighbor_table: Dict[int, List[int]],
    K: CameraIntrinsics,
    C_CB: np.ndarray,
    max_iters: int = 3,
    debug: bool = False
) -> Tuple[Dict[int, Tuple[float,float]], QuatSL]:
    """
    Returns mapping: cat_index -> predicted (u,v) at n+1, and q_pred.
    Seeds: stars identified at n. If none found, FALL BACK to projecting the entire catalog.
    """
    id_to_idx = {s["id"]: i for i, s in enumerate(catalog)}

    # Predict attitude at n+1 from (n-1,n)
    omega = angular_velocity_from_two_quats(q_nm1, q_n, dt)   # rad/s
    q_pred = predict_next_quat(q_n, omega, dt)
    C_BI = q_to_dcm_BI(q_pred)

    # Try to seed from known IDs
    queue: deque[int] = deque()
    visited: Set[int] = set()
    pred_uv: Dict[int, Tuple[float,float]] = {}

    provided = list(known_ids_at_n)
    seeds_found = 0
    for sid in provided:
        idx = id_to_idx.get(sid, None)
        if idx is not None:
            queue.append(idx)
            seeds_found += 1
    if debug:
        print(f"[seed] provided={len(provided)}, found_in_catalog={seeds_found}")

    if seeds_found == 0:
        # FALLBACK: Project entire catalog once; keep stars that land in FOV
        for i, s in enumerate(catalog):
            uv = inertial_to_pixel(np.asarray(s["vec"], float), C_BI, C_CB, K)
            if uv is not None:
                pred_uv[i] = uv
        if debug:
            print(f"[predict] (fallback: whole-catalog) in-FOV predictions: {len(pred_uv)}")
        return pred_uv, q_pred

    # BFS growth from seeds (neighbors ≤9)
    it = 0
    while queue and it < max_iters:
        layer_count = len(queue)
        for _ in range(layer_count):
            idx = queue.popleft()
            if idx in visited:
                continue
            visited.add(idx)

            uv = inertial_to_pixel(np.asarray(catalog[idx]["vec"], float), C_BI, C_CB, K)
            if uv is None:
                continue
            pred_uv[idx] = uv

            for nb in neighbor_table.get(idx, []):
                if nb not in visited:
                    queue.append(nb)
        it += 1

    if debug:
        print(f"[predict] predicted {len(pred_uv)} star positions (within FOV).")
    return pred_uv, q_pred

# =============================================================================
# 6) Match predicted pixels to measured centroids (d = tracking_radius_px)
# =============================================================================
def match_predicted_to_centroids(
    pred_uv: Dict[int, Tuple[float,float]],
    centroids: np.ndarray,
    radius_px: float = 5.0,
    debug: bool = False
) -> Dict[int, int]:
    if centroids.size == 0 or len(pred_uv) == 0:
        if debug:
            print(f"[match] skipped (centroids={centroids.size}, predictions={len(pred_uv)})")
        return {}

    assigned: Dict[int, int] = {}
    used_centroids: Set[int] = set()
    r2 = radius_px * radius_px

    items = list(pred_uv.items())
    # sort by their nearest-possible centroid distance (greedy + stable)
    dmins = []
    for idx, (u_pred, v_pred) in items:
        diffs = centroids - np.array([u_pred, v_pred])[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        dmins.append((idx, float(np.min(d2))))
    order = [i for i,_ in sorted(dmins, key=lambda t: t[1])]

    for idx in order:
        u_pred, v_pred = pred_uv[idx]
        diffs = centroids - np.array([u_pred, v_pred])[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        k = int(np.argmin(d2))
        if d2[k] <= r2 and k not in used_centroids:
            assigned[idx] = k
            used_centroids.add(k)

    if debug:
        print(f"[match] matched {len(assigned)} / {len(pred_uv)} within {radius_px:.1f}px")
    return assigned

# =============================================================================
# 7) QUEST solve
# =============================================================================
def quest_attitude_from_matches(
    matches: Dict[int, int],
    catalog: List[dict],
    centroids: np.ndarray,
    K: CameraIntrinsics,
    C_CB: np.ndarray,
    min_vectors: int
) -> Optional[QuatSL]:
    if len(matches) < min_vectors:
        return None

    body_vecs: List[Vec3] = []
    inertial_vecs: List[Vec3] = []

    for cat_idx, c_idx in matches.items():
        u, v = centroids[c_idx]
        b_body = pixel_to_body_ray(u, v, K, C_CB)
        b_body = b_body / (np.linalg.norm(b_body) + 1e-15)
        r_i = np.asarray(catalog[cat_idx]["vec"], float)
        r_i = r_i / (np.linalg.norm(r_i) + 1e-15)
        body_vecs.append(b_body)
        inertial_vecs.append(r_i)

    q_sf, _, _ = QUEST.quest_method(
        body_vectors=np.asarray(body_vecs, float),
        inertial_vectors=np.asarray(inertial_vecs, float),
        weights=None
    )
    q_sl_B2I = np.array([q_sf[1], q_sf[2], q_sf[3], q_sf[0]], float)
    q_sl_I2B = q_conj(q_sl_B2I)
    return q_norm(q_sl_I2B)

# =============================================================================
# 8) End-to-end tracking step
# =============================================================================
def tracking_step(
    q_nm1_sl: QuatSL,                # q at n-1 (I->B, scalar-last) or B->I if cfg.quats_are_I2B=False
    q_n_sl: QuatSL,                  # q at n   (I->B, scalar-last) or B->I if cfg.quats_are_I2B=False
    dt: float,
    known_ids_at_n: Iterable[str],
    image_np1_path: str,
    catalog_path: str,
    fov_deg_vertical: float,
    C_CB: Optional[np.ndarray] = None,
    cfg: TrackConfig = TrackConfig()
) -> Dict[str, object]:

    # Conventions: flip if the inputs are actually B->I
    if not cfg.quats_are_I2B:
        q_nm1_sl = q_conj(q_nm1_sl)
        q_n_sl   = q_conj(q_n_sl)

    # 1) Centroids
    stars, img_w, img_h = detect_stars(image_np1_path, threshold_rel=0.60, min_area=2, dedupe_px=2.0)
    centroids = np.array([[s["x"], s["y"]] for s in stars], float)
    if cfg.debug:
        print(f"[detect] centroids: {len(centroids)}  (w,h)=({img_w},{img_h})")

    # 2) Intrinsics from vertical FOV
    f_pix, cx, cy = intrinsics_from_fov(img_w, img_h, fov_deg_vertical, vertical=True)
    K = CameraIntrinsics(f_pix=f_pix, cx=cx, cy=cy, width=img_w, height=img_h)
    if cfg.debug:
        fov_rad = math.radians(fov_deg_vertical)
        px_shift_0p05deg = (math.radians(0.05) * img_h) / fov_rad
        print(f"[intrinsics] f_pix={f_pix:.2f}  cx,cy=({cx:.1f},{cy:.1f})  ~Δpix(0.05°)≈{px_shift_0p05deg:.2f}px")

    # 3) Catalog
    catalog = parse_catalog(catalog_path, mag_limit=cfg.mag_limit)
    if cfg.debug:
        print(f"[catalog] count={len(catalog)} (V ≤ {cfg.mag_limit})")

    # 4) Neighbor table
    neighbor_table = build_neighbor_catalog(catalog, k=cfg.neighbors_per_star)

    # 5) Camera-from-Body DCM
    if C_CB is None:
        C_CB = np.eye(3)

    # 6) Predict star pixels (with robust fallback) + get q_pred (model-only)
    pred_uv, q_pred = predict_star_pixels_next(
        q_nm1=q_norm(q_nm1_sl),
        q_n=q_norm(q_n_sl),
        dt=dt,
        known_ids_at_n=known_ids_at_n,
        catalog=catalog,
        neighbor_table=neighbor_table,
        K=K,
        C_CB=C_CB,
        max_iters=4,
        debug=cfg.debug
    )

    # 7) Match
    assignments = match_predicted_to_centroids(
        pred_uv=pred_uv,
        centroids=centroids,
        radius_px=cfg.tracking_radius_px,
        debug=cfg.debug
    )

    # 8) QUEST (or fallback to q_pred if not enough matches)
    q_np1_est = quest_attitude_from_matches(
        matches=assignments,
        catalog=catalog,
        centroids=centroids,
        K=K,
        C_CB=C_CB,
        min_vectors=cfg.min_vectors_for_quest
    )
    used_fallback = False
    if q_np1_est is None:
        # Ensure you always get a quaternion to compare
        q_np1_est = q_norm(q_pred)
        used_fallback = True
        if cfg.debug:
            print("[quest] not enough pairs — returning model-only q_pred as fallback")

    return {
        "q_np1_est_sl": q_np1_est,
        "model_only_q_pred": q_norm(q_pred),
        "used_model_fallback": used_fallback,
        "num_tracked": len(assignments),
        "assignments": assignments,    # cat_idx -> centroid_idx
        "predicted_uv": pred_uv,       # cat_idx -> (u,v)
        "centroids": centroids,
        "K": K,
        "C_CB": C_CB
    }

# =============================================================================
# 9) Small helper to build C_CB from a LiS variant (optional)
# =============================================================================
def build_C_CB_from_variant(perm: Tuple[int,int,int], signs: Tuple[int,int,int]) -> np.ndarray:
    P = np.zeros((3,3))
    for r,c in enumerate(perm):
        P[r,c] = 1.0
    S = np.diag(list(signs))
    return S @ P

# =============================================================================
# 10) Example main (FILL THE TODOs with your data)
# =============================================================================
if __name__ == "__main__":
    # >>> Two consecutive LiS quaternions; set quats_are_I2B in cfg accordingly <<<
    q_nm1 = np.array([ 0.70624133, 0.54916346, -0.44337591, 0.05532166], float)  # step n-1
    q_n   = np.array([ 0.706469, 0.548894, -0.443381, 0.055045], float)  # step n

    DELTA_T = 1.0

    known_ids_at_n = ["HIP69673", "HIP72105", "HIP76267", "HIP77070", "HIP81377", "HIP72622", "HIP79593", "HIP74785"]

    image_np1_path = r"C:\Users\mubas\OneDrive\Documents\Modules\Year 1\Endurosat\SPARCS\src\trial_0.05_3.png"
    catalog_path   = r"C:\Users\mubas\OneDrive\Documents\Modules\Year 1\Endurosat\SPARCS\src\HipparcosCatalog.txt"
    FOV_DEG_VERTICAL = 52.3

    # Set C_CB correctly. If LiS “winner” variant was perm(0,2,1)_sign(-1,1,1):
    # C_CB = build_C_CB_from_variant((0,2,1), (-1,1,1))
    C_CB = np.eye(3)

    cfg = TrackConfig(
        neighbors_per_star=9,
        tracking_radius_px=5.0,
        min_vectors_for_quest=3,
        mag_limit=4.5,
        quats_are_I2B=True,  # set False if your inputs are actually B->I
        debug=True
    )

    out = tracking_step(
        q_nm1_sl=q_nm1,
        q_n_sl=q_n,
        dt=DELTA_T,
        known_ids_at_n=known_ids_at_n,
        image_np1_path=image_np1_path,
        catalog_path=catalog_path,
        fov_deg_vertical=FOV_DEG_VERTICAL,
        C_CB=C_CB,
        cfg=cfg
    )

    q_est = out["q_np1_est_sl"]
    q_pred = out["model_only_q_pred"]

    print("\n=== Tracking result (n+1) ===")
    print("Tracked stars:", out["num_tracked"])
    print("Used model-only fallback:", out["used_model_fallback"])
    print("Estimated q_{n+1} (I->B, scalar-last):", np.array2string(q_est, precision=8))
    print("Model-only q_pred (I->B, scalar-last):", np.array2string(q_pred, precision=8))
    print("Estimated q_{n+1} (I->B, scalar-last):", None if q_est is None else np.array2string(q_est, precision=8))
# Angular comparisons (only if we got an estimate)
if q_est is not None:
    ang_nm1_to_n   = quaternion_angular_distance(q_nm1, q_n)
    ang_n_to_est   = quaternion_angular_distance(q_n,   q_est)
    ang_nm1_to_est = quaternion_angular_distance(q_nm1, q_est)

    print("\n=== Angular comparisons (deg) ===")
    print(f"angle(q_(n-1), q_n)          : {ang_nm1_to_n:.6f}")
    print(f"angle(q_n, q_(n+1)_est)      : {ang_n_to_est:.6f}")
    print(f"angle(q_(n-1), q_(n+1)_est)  : {ang_nm1_to_est:.6f}")
else:
    print("\n(No q_(n+1) estimate — skipping angular comparisons.)")

