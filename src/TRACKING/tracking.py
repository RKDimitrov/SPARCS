##TRACKING CODE TESTING USING JAPANESE METHOD
#Quaternion: [0.370812, 0.114763, 0.047753, 0.920352]
#Quaternion: [0.386422, -0.035837, -0.308044, 0.868621]

# N approach_2.py
# Tracking-mode single step:
# - hard-codes q1 from Lost-in-Space
# - predicts/star-matches on image 2
# - re-estimates attitude with your QUEST
# - prints q2_est (scalar-last [qx,qy,qz,qw], inertial->body)

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import numpy as np
import math
import importlib.util, sys

# =============================================================================
# Load your Lost-in-Space utilities from the other folder
# =============================================================================
module_path = r"C:\Users\mubas\OneDrive\Documents\Modules\Year 1\Endurosat\SPARCS\src\NEW ALTERNATE APPROACH\N approach_2.py"
module_name = "n_approach_2"

spec = importlib.util.spec_from_file_location(module_name, module_path)
lis = importlib.util.module_from_spec(spec)
sys.modules[module_name] = lis
spec.loader.exec_module(lis)

# LiS helpers
QUEST = lis.QUEST
detect_stars = lis.detect_stars
parse_catalog = lis.parse_catalog
intrinsics_from_fov = lis.intrinsics_from_fov
pixel_to_ray_stereo_f = lis.pixel_to_ray_stereo_f
ray_to_pixel_stereo_f = lis.ray_to_pixel_stereo_f
flip_matrix = lis.flip_matrix

# =============================== TYPES/CONFIG =================================

Vec3 = np.ndarray
QuatSL = np.ndarray  # scalar-last [qx,qy,qz,qw]

@dataclass
class CameraIntrinsics:
    f_pix: float
    cx: float
    cy: float
    width: int
    height: int

@dataclass
class TrackingConfig:
    # Your camera uses a VERTICAL FOV. We force vertical axis only.
    # Angle-space gates (deg), from loose to tight:
    angle_gate_degs: Tuple[float, ...] = (1.0, 0.6, 0.3)
    # Inlier refinement gates (deg): after QUEST, drop pairs above these and re-solve
    refine_gates_deg: Tuple[float, ...] = (0.3, 0.15)
    # Minimum vectors to run QUEST
    min_vectors_for_quest: int = 3
    # Pixel fallback (can introduce bad pairs for tiny motions). Disable by default.
    use_pixel_fallback: bool = False
    match_radius_px: float = 20.0  # only used if use_pixel_fallback=True
    # Try both q1 meanings (safety): I->B (as given) and B->I (conjugate)
    try_q1_directions: Tuple[str, ...] = ("I2B", "B2I")
    # Flip variants to try if not forced
    variants: Tuple[str, ...] = ("normal","flip_x","flip_y","flip_xy")
    # Optionally force a specific flip variant if you know what LiS used (e.g. "flip_x")
    force_variant: Optional[str] = None
    # Catalogue brightness limit (more stars -> easier matching)
    mag_limit: float = 4.5

# ============================= QUATERNION UTILS ===============================

def q_sl_normalize(q: QuatSL) -> QuatSL:
    q = np.asarray(q, float); n = np.linalg.norm(q)
    return q if n == 0 else q / n

def q_sl_conj(q: QuatSL) -> QuatSL:
    return np.array([-q[0], -q[1], -q[2], q[3]], float)

def q_sl_mul(q2: QuatSL, q1: QuatSL) -> QuatSL:
    # Hamilton product, scalar-last
    x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
    v = np.array([
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ])
    w = w2*w1 - (x2*x1 + y2*y1 + z2*z1)
    return q_sl_normalize(np.array([*v, w], float))

def rotmat_BI_from_q_sl(q_BI: QuatSL) -> np.ndarray:
    x, y, z, w = q_sl_normalize(q_BI)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)]
    ], float)

def q_sf_from_q_sl(q_sl: QuatSL) -> np.ndarray:
    return np.array([q_sl[3], q_sl[0], q_sl[1], q_sl[2]], float)

def q_sl_from_q_sf(q_sf: np.ndarray) -> np.ndarray:
    """[qw,qx,qy,qz] -> [qx,qy,qz,qw] (scalar-last)."""
    q_sf = np.asarray(q_sf, float)
    return np.array([q_sf[1], q_sf[2], q_sf[3], q_sf[0]], float)

def quat_angle_deg(q1_sl: QuatSL, q2_sl: QuatSL) -> float:
    q1 = q_sl_normalize(q1_sl); q2 = q_sl_normalize(q2_sl)
    dot = float(np.clip(abs(q1 @ q2), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))

# ============================ CAMERA PROJECTION ===============================

def project_body_ray_to_pixel(b_unit: Vec3, K: CameraIntrinsics, variant: str) -> Optional[Tuple[float,float]]:
    F = flip_matrix(variant)
    x, y, ok = ray_to_pixel_stereo_f(F @ b_unit, K.cx, K.cy, K.f_pix)
    if not ok or x < 0 or x >= K.width or y < 0 or y >= K.height:
        return None
    return (float(x), float(y))

def pixel_to_body_ray(u: float, v: float, K: CameraIntrinsics, variant: str) -> Vec3:
    F = flip_matrix(variant)
    return F @ pixel_to_ray_stereo_f(u, v, K.cx, K.cy, K.f_pix)

# ================================ QUEST WRAP ==================================

def quest_from_pairs(body_vecs: List[np.ndarray], inertial_vecs: List[np.ndarray]) -> np.ndarray:
    """
    Calls your QUEST implementation.
    Input lists are arrays of unit vectors (body/inertial).
    Returns scalar-last quaternion (inertial->body).
    """
    q_sf, _, _ = QUEST.quest_method(
        body_vectors=np.asarray(body_vecs, float),
        inertial_vectors=np.asarray(inertial_vecs, float),
        weights=None
    )  # LiS returns scalar-first [w,x,y,z] for Body->Inertial
    q_sl_B2I = q_sl_from_q_sf(q_sf)   # scalar-last, Body->Inertial
    q_sl_I2B = q_sl_conj(q_sl_B2I)    # convert to Inertial->Body
    return q_sl_normalize(q_sl_I2B)

# ============================ MATCHING HELPERS ================================

def angle_match(
    C_BI_pred: np.ndarray,
    K: CameraIntrinsics,
    centroids_uv: np.ndarray,
    catalog_vecs: np.ndarray,
    variant: str,
    angle_gate_deg: float
) -> List[Tuple[int,int]]:
    """
    Back-project centroid -> body ray (with F), rotate to inertial (C_BI_pred^T),
    pick nearest catalog star by angular separation within angle_gate_deg.
    Returns list of (measured_idx, catalog_subset_idx).
    """
    if centroids_uv.size == 0 or catalog_vecs.size == 0:
        return []
    C_IB_pred = C_BI_pred.T
    matches = []
    used = set()
    for j, (u, v) in enumerate(centroids_uv):
        b_meas = pixel_to_body_ray(u, v, K, variant)
        r_est = C_IB_pred @ b_meas
        r_est /= (np.linalg.norm(r_est) + 1e-15)
        dots = catalog_vecs @ r_est
        k = int(np.argmax(dots))
        ang = math.degrees(math.acos(float(np.clip(dots[k], -1.0, 1.0))))
        if ang <= angle_gate_deg and k not in used:
            used.add(k)
            matches.append((j, k))
    return matches

def angular_residual_deg(C_BI: np.ndarray, b_meas: Vec3, r_inertial: Vec3) -> float:
    """
    Angle between catalog inertial direction and the inertial ray reconstructed
    from a measured body ray under attitude C_BI:  r_hat = C_IB @ b_meas.
    """
    C_IB = C_BI.T
    r_hat = C_IB @ b_meas
    r_hat /= (np.linalg.norm(r_hat) + 1e-15)
    r = r_inertial / (np.linalg.norm(r_inertial) + 1e-15)
    dot = float(np.clip(np.dot(r_hat, r), -1.0, 1.0))
    return math.degrees(math.acos(dot))

# ================================ TRACKING ====================================

def tracking_mode(
    q1_sl_input: QuatSL,                # hard-coded q1 (scalar-last if direction='I2B')
    catalog_list: List[dict],           # parse_catalog(...)
    K_vert: CameraIntrinsics,           # intrinsics from vertical FOV (forced)
    K_horz: Optional[CameraIntrinsics], # unused (kept for signature compatibility)
    centroids_img2: np.ndarray,         # (M,2)
    delta_t: float = 1.0,
    omega_body: Optional[Vec3] = None,
    cfg: TrackingConfig = TrackingConfig(),
    fov_deg_vertical: float = 52.3,
    fov_deg_horizontal: Optional[float] = None,
    seed_star_ids: Optional[Iterable[str]] = None   # optional: restrict to stars tracked in frame 1
):
    # Prepare catalogue arrays
    cat_vecs_all = np.array([s["vec"] for s in catalog_list], float)
    cat_ids_all  = [s["id"] for s in catalog_list]
    if seed_star_ids is not None:
        seed_set = set(seed_star_ids)
        keep_mask = np.array([sid in seed_set for sid in cat_ids_all], bool)
        if not np.any(keep_mask):
            keep_mask = np.ones(len(cat_ids_all), bool)  # fall back if seed list empty/mismatched
        cat_vecs_all = cat_vecs_all[keep_mask]
        cat_ids_all  = [sid for sid, km in zip(cat_ids_all, keep_mask) if km]

    # Best-so-far accumulator
    best = {
        "q2_est_sl": None,
        "num_matched": -1,
        "rms": None,
        "variant": None,
        "axis_used": "vertical",  # forced
        "q1_mode": None
    }

    # Force vertical FOV
    K = K_vert
    half_fov = 0.5 * fov_deg_vertical

    # Which flip variants to try
    variants_to_try = (cfg.force_variant,) if cfg.force_variant else cfg.variants

    # Try both interpretations of q1 (safety)
    for q1_mode in cfg.try_q1_directions:
        q1_sl = q_sl_normalize(q1_sl_input if q1_mode == "I2B" else q_sl_conj(q1_sl_input))

        # Motion model (constant ω). For no IMU, identity increment.
        if omega_body is None or float(np.linalg.norm(omega_body)) < 1e-12:
            q_delta = np.array([0.0,0.0,0.0,1.0], float)
        else:
            theta = float(np.linalg.norm(omega_body) * delta_t)
            u = omega_body / (np.linalg.norm(omega_body) + 1e-15)
            s = math.sin(theta/2.0)
            q_delta = q_sl_normalize(np.array([u[0]*s, u[1]*s, u[2]*s, math.cos(theta/2.0)], float))

        q_pred_sl = q_sl_mul(q_delta, q1_sl)
        C_BI_pred = rotmat_BI_from_q_sl(q_pred_sl)

        for variant in variants_to_try:
            F = flip_matrix(variant)

            # Predict which catalogue stars are inside the FOV using angle from boresight
            b_cam = (F @ (C_BI_pred @ cat_vecs_all.T)).T  # inertial->body->camera
            bz = b_cam[:,2]
            ang_from_bore = np.degrees(np.arccos(np.clip(bz, -1.0, 1.0)))
            keep = ang_from_bore <= (half_fov * 1.05)  # small slack
            cat_subset = cat_vecs_all[keep]
            if cat_subset.size == 0:
                continue

            # Sweep angular gates from tight to tighter (avoid wrong pairs)
            chosen_pairs = None
            for gate in cfg.angle_gate_degs:
                pairs = angle_match(C_BI_pred, K, centroids_img2, cat_subset, variant, gate)
                if len(pairs) >= cfg.min_vectors_for_quest:
                    chosen_pairs = pairs
                    break

            body_vecs: List[np.ndarray] = []
            inertial_vecs: List[np.ndarray] = []
            pairs_for_rms: List[Tuple[float,float,np.ndarray]] = []

            # ANGLE-SPACE pairs (preferred)
            if chosen_pairs:
                for j_meas, k_sub in chosen_pairs:
                    u, v = centroids_img2[j_meas]
                    b_meas = pixel_to_body_ray(u, v, K, variant)
                    b_meas /= (np.linalg.norm(b_meas) + 1e-15)
                    r_i = cat_subset[k_sub]

                    body_vecs.append(b_meas)
                    inertial_vecs.append(r_i)
                    pairs_for_rms.append((u, v, r_i))

            # PIXEL fallback (optional; disabled by default for tiny motions)
            if cfg.use_pixel_fallback and len(body_vecs) < cfg.min_vectors_for_quest:
                proj_xy = []
                for r_i in cat_subset:
                    b = C_BI_pred @ r_i
                    b /= (np.linalg.norm(b) + 1e-15)
                    px = project_body_ray_to_pixel(b, K, variant)
                    if px is not None:
                        proj_xy.append((r_i, px))
                if proj_xy:
                    pred_xy = np.array([p for _, p in proj_xy], float)
                    used = set()
                    for (u, v) in centroids_img2:
                        diffs = pred_xy - np.array([u, v])
                        d2 = np.einsum('ij,ij->i', diffs, diffs)
                        k = int(np.argmin(d2))
                        if k in used or d2[k] > cfg.match_radius_px**2:
                            continue
                        used.add(k)
                        r_i = proj_xy[k][0]
                        b_meas = pixel_to_body_ray(u, v, K, variant)
                        b_meas /= (np.linalg.norm(b_meas) + 1e-15)
                        body_vecs.append(b_meas)
                        inertial_vecs.append(r_i)
                        pairs_for_rms.append((u, v, r_i))

            # Enough pairs to solve?
            if len(body_vecs) < cfg.min_vectors_for_quest:
                continue

            # Initial QUEST solution (convert B->I to I->B inside quest_from_pairs)
            q_est = quest_from_pairs(body_vecs, inertial_vecs)
            if (q_est @ q_pred_sl) < 0:
                q_est = -q_est

            # Inlier refinement loop (drop pairs with big angular residuals and re-solve)
            for gate_ref in cfg.refine_gates_deg:
                C_est = rotmat_BI_from_q_sl(q_est)
                keep_idx = []
                for idx, (u, v, r_i) in enumerate(pairs_for_rms):
                    b_meas = pixel_to_body_ray(u, v, K, variant)
                    b_meas /= (np.linalg.norm(b_meas) + 1e-15)
                    res = angular_residual_deg(C_est, b_meas, r_i)
                    if res <= gate_ref:
                        keep_idx.append(idx)

                if len(keep_idx) >= cfg.min_vectors_for_quest:
                    body_vecs = [body_vecs[i] for i in keep_idx]
                    inertial_vecs = [inertial_vecs[i] for i in keep_idx]
                    pairs_for_rms = [pairs_for_rms[i] for i in keep_idx]
                    q_est = quest_from_pairs(body_vecs, inertial_vecs)
                    if (q_est @ q_pred_sl) < 0:
                        q_est = -q_est
                else:
                    break

            # Score by reprojection RMS on actual pairs, prioritising lowest RMS first
            C_est = rotmat_BI_from_q_sl(q_est)
            errs = []
            for (u, v, r_i) in pairs_for_rms:
                bproj = C_est @ (r_i / (np.linalg.norm(r_i) + 1e-15))
                bproj /= (np.linalg.norm(bproj) + 1e-15)
                px = project_body_ray_to_pixel(bproj, K, variant)
                if px is None:
                    continue
                errs.append(math.hypot(px[0] - u, px[1] - v))
            rms = float(np.sqrt(np.mean(np.array(errs)**2))) if errs else None
            num_matched = len(pairs_for_rms)
            effective_rms = rms if rms is not None else 1e9
            score = (-effective_rms, num_matched)  # prioritise lowest RMS, then more matches

            best_score = (-(best["rms"] if best["rms"] is not None else 1e9), best["num_matched"])
            if score > best_score:
                best.update({
                    "q2_est_sl": q_est,
                    "num_matched": num_matched,
                    "rms": rms,
                    "variant": variant,
                    "q1_mode": q1_mode
                })

    # Fallback if nothing worked
    if best["q2_est_sl"] is None:
        q1_default = q_sl_normalize(q1_sl_input)
        best["q2_est_sl"] = q1_default
        best["num_matched"] = 0
        best["rms"] = None
        best["variant"] = None
        best["q1_mode"] = None

    return best

# ================================== RUN =======================================

if __name__ == "__main__":
    # Paths
    image2_png = r"C:\Users\mubas\OneDrive\Documents\Modules\Year 1\Endurosat\SPARCS\src\pov2_Track_0.05.png"
    catalog_csv = r"C:\Users\mubas\OneDrive\Documents\Modules\Year 1\Endurosat\SPARCS\src\HipparcosCatalog.txt"

    # q1 for image 1 (scalar-last, inertial->body)
    q1_sl_input = np.array([0.37916563, 0.79160741, -0.03189190, -0.47809418], float)

    # Detect centroids in image 2
    stars_img2, img_w, img_h = detect_stars(image2_png, threshold_rel=0.60, min_area=2, dedupe_px=2.0)
    centroids = np.array([[s["x"], s["y"]] for s in stars_img2], float)
    print(f"[detect_stars] Found {len(centroids)} candidates.")

    # Intrinsics: VERTICAL FOV ONLY (forced)
    FOV_DEG = 52.3
    f_v, cx_v, cy_v = intrinsics_from_fov(img_w, img_h, FOV_DEG, vertical=True)
    K_vert = CameraIntrinsics(f_pix=f_v, cx=cx_v, cy=cy_v, width=img_w, height=img_h)

    # Catalog
    CATALOG_MAG_LIMIT = 4.5
    catalog = parse_catalog(catalog_csv, mag_limit=CATALOG_MAG_LIMIT)
    print(f"[parse_catalog] Parsed {len(catalog)} stars (V ≤ {CATALOG_MAG_LIMIT}).")

    # Configure tracker:
    # - Force vertical FOV (done internally)
    # - Tight angle and refine gates
    # - Disable pixel fallback for tiny motion
    # - Optionally force variant to the one LiS used (e.g., "flip_x")
    cfg = TrackingConfig(
        angle_gate_degs=(1.0, 0.6, 0.3),
        refine_gates_deg=(0.3, 0.15),
        use_pixel_fallback=False,
        match_radius_px=20.0,
        try_q1_directions=("I2B", "B2I"),
        variants=("normal","flip_x","flip_y","flip_xy"),
        force_variant=None,   # set to "flip_x" if you know LiS used it
        mag_limit=CATALOG_MAG_LIMIT
    )

    out = tracking_mode(
        q1_sl_input=q1_sl_input,
        catalog_list=catalog,
        K_vert=K_vert,
        K_horz=None,                # unused (kept for signature compatibility)
        centroids_img2=centroids,
        delta_t=1.0,
        omega_body=None,            # or provide gyro rate in rad/s
        cfg=cfg,
        fov_deg_vertical=FOV_DEG,
        fov_deg_horizontal=None,
        seed_star_ids=None          # optionally pass IDs from frame 1 to constrain matching
    )

    q2_est = out["q2_est_sl"]
    print("\n=== Tracking Mode Result (Image 2) ===")
    print(f"Chosen flip variant: {out['variant']}")
    print(f"FOV axis used: vertical (forced)")
    print(f"q1 interpretation used: {out['q1_mode']}")
    print("q1 (scalar-last [qx,qy,qz,qw], I->B):")
    print(np.array2string(q1_sl_input, precision=8, floatmode='fixed'))
    print("q2_est (scalar-last [qx,qy,qz,qw], I->B):")
    print(np.array2string(q2_est, precision=8, floatmode='fixed'))

    print("\nAngular distance (deg):")
    print("  angle(q1, q2_est):", f"{quat_angle_deg(q1_sl_input, q2_est):.6f}")

    print("\ntracking stats:",
          {"num_matched": out["num_matched"], "rms_reproj_px": out["rms"], "chosen_variant": out["variant"]})






