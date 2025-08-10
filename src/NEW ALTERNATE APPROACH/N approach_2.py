import csv
import math
import random
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# ================ QUEST IMPLEMENTATION ================

def adjoint_matrix(M):
    return np.linalg.det(M) * np.linalg.inv(M)

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

        for iteration in range(max_iter):
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
        return q, lmbd, iteration + 1

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

# ================ ORIGINAL STAR IDENTIFICATION CODE ================

def parse_ra_to_deg(ra_val):
    if isinstance(ra_val, (int, float)):
        v = float(ra_val)
        return v*15.0 if v <= 24.0 else (v % 360.0)
    s = str(ra_val).strip().replace(":", " ").replace("h"," ").replace("m"," ").replace("s"," ")
    p = [q for q in s.split() if q]
    if len(p) == 3:
        hh, mm, ss = map(float, p)
        return (hh + mm/60.0 + ss/3600.0) * 15.0
    v = float(s)
    return v*15.0 if v <= 24.0 else (v % 360.0)

def parse_dec_to_deg(dec_val):
    if isinstance(dec_val, (int, float)): return float(dec_val)
    s = str(dec_val).strip().replace(":", " ").replace("d"," ").replace("m"," ").replace("s"," ")
    p = [q for q in s.split() if q]
    if len(p) == 3:
        sign = -1.0 if p[0].startswith("-") else 1.0
        dd = abs(float(p[0])); mm = float(p[1]); ss = float(p[2])
        return sign * (dd + mm/60.0 + ss/3600.0)
    return float(s)

def unit_vec_from_radec(ra_deg, dec_deg):
    ra = math.radians(ra_deg); dec = math.radians(dec_deg)
    return np.array([math.cos(dec)*math.cos(ra),
                     math.cos(dec)*math.sin(ra),
                     math.sin(dec)], float)

def sniff_delimiter(line):
    if "|" in line: return "|"
    if "," in line: return ","
    if "\t" in line: return "\t"
    return None

def parse_catalog(file_path, mag_limit=3):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        if not head: raise ValueError("Empty catalogue.")
        delim = sniff_delimiter(head); f.seek(0)
        rows=[]
        if delim is None:
            for i,line in enumerate(f):
                line=line.strip()
                if not line: continue
                if i==0: rows.append([h.strip().lower() for h in line.split()])
                else: rows.append(line.split())
        else:
            rdr = csv.reader(f, delimiter=delim)
            rows = [row for row in rdr if row and any(cell.strip() for cell in row)]
    header=[h.strip().lower() for h in rows[0]]
    def idx(col,alts):
        if col in header: return header.index(col)
        for a in alts:
            if a in header: return header.index(a)
        raise ValueError(f"Missing column like '{col}' in {header}")
    i_name=idx("name",["id","hip","hip_id","hipparcos","hipparcos_id"])
    i_ra=idx("ra",["ra_deg","ra_h","right_ascension","raj2000","ra2000"])
    i_dec=idx("dec",["dec_deg","decl","declination","dej2000","dec2000"])
    i_mag=idx("vmag",["mag","v","vmag_v","appmag","m_v"])

    stars=[]; bad=0
    for r in rows[1:]:
        try:
            name=r[i_name].strip()
            vmag=float(r[i_mag])
            if vmag>mag_limit: continue
            ra_deg=parse_ra_to_deg(r[i_ra]); dec_deg=parse_dec_to_deg(r[i_dec])
            vec=unit_vec_from_radec(ra_deg, dec_deg)
            if not np.isfinite(vec).all(): continue
            stars.append({"id":name,"ra_deg":ra_deg,"dec_deg":dec_deg,"vmag":vmag,"vec":vec})
        except Exception:
            bad+=1
    if not stars: raise ValueError("Parsed 0 stars.")
    if bad: print(f"[parse_catalog] Skipped {bad} malformed rows.")
    print(f"[parse_catalog] Parsed {len(stars)} stars (V ≤ {mag_limit}).")
    return stars

def detect_stars(image_path, threshold_rel=0.60, min_area=2, dedupe_px=2.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(image_path)
    h,w = img.shape[:2]
    blurred = cv2.GaussianBlur(img,(3,3),0)
    thr_val = max(5.0, float(blurred.max())*threshold_rel)
    _, bw = cv2.threshold(blurred, thr_val, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars=[]
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area: continue
        M=cv2.moments(cnt); 
        if M["m00"]==0: continue
        cx_= M["m10"]/M["m00"]; cy_ = M["m01"]/M["m00"]
        mask = np.zeros_like(img, np.uint8); cv2.drawContours(mask,[cnt],-1,255,-1)
        brightness = float(img[mask==255].sum())
        stars.append({"x":cx_,"y":cy_,"brightness":brightness})
    stars.sort(key=lambda s:s["brightness"], reverse=True)
    kept=[]
    for s in stars:
        if all((abs(s["x"]-t["x"])>dedupe_px or abs(s["y"]-t["y"])>dedupe_px) for t in kept):
            kept.append(s)
    print(f"[detect_stars] Found {len(kept)} candidates.")
    return kept, w, h

def pixel_to_ray_stereo_f(x, y, cx, cy, f_pix):
    Xp = (x - cx); Yp = (cy - y)
    u = Xp/(2.0*f_pix); v = Yp/(2.0*f_pix)
    denom = 1.0 + u*u + v*v
    vx = 2.0*u/denom; vy = 2.0*v/denom; vz = (1.0 - u*u - v*v)/denom
    v3 = np.array([vx,vy,vz], float)
    v3 /= (np.linalg.norm(v3)+1e-12)
    return v3

def ray_to_pixel_stereo_f(v_cam, cx, cy, f_pix):
    vx,vy,vz = float(v_cam[0]), float(v_cam[1]), float(v_cam[2])
    if vz <= 1e-12: return None, None, False
    Xp = 2.0*f_pix * vx/(1.0+vz)
    Yp = 2.0*f_pix * vy/(1.0+vz)
    x = cx + Xp; y = cy - Yp
    return x, y, True

def angle_between(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v)))))

def kabsch_cam_to_world(A_cam, B_world):
    H = A_cam.T @ B_world
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2] *= -1; R = Vt.T @ U.T
    return R

def flip_matrix(variant):
    if variant == "flip_x":  return np.diag([-1.0, 1.0, 1.0])
    if variant == "flip_y":  return np.diag([ 1.0,-1.0, 1.0])
    if variant == "flip_xy": return np.diag([-1.0,-1.0, 1.0])
    return np.eye(3)

def build_catalog_descriptors(catalog, k=5, neighbor_max_deg=70.0):
    coords = np.array([s["vec"] for s in catalog], float)
    tree = cKDTree(coords)
    desc = []; neigh = []
    for idx in range(len(catalog)):
        d, j = tree.query(coords[idx], k=k+1)
        j = j[1:]
        th = [angle_between(coords[idx], coords[int(t)]) for t in j]
        keep = sorted([(t,int(jj)) for t,jj in zip(th,j)], key=lambda z:z[0])
        keep = [kv for kv in keep if kv[0] <= neighbor_max_deg]
        if len(keep) < k: keep = keep[:k]
        keep = keep[:k]
        th = [kv[0] for kv in keep]
        j  = [kv[1] for kv in keep]
        if len(th) < k:
            th += [th[-1]]*(k-len(th)); j += [j[-1]]*(k-len(j))
        tanh = [math.tan(math.radians(t)/2.0) for t in th]
        tanh[-1] = tanh[-1] if tanh[-1] != 0 else 1e-6
        ratios = [tanh[i]/tanh[-1] for i in range(k-1)]
        desc.append(ratios); neigh.append(j)
    return np.array(desc, float), neigh, coords

def build_image_descriptors(stars, k=5):
    pts = np.array([[s["x"], s["y"]] for s in stars], float)
    tree = cKDTree(pts)
    desc = []; neigh = []
    for i in range(len(stars)):
        d, j = tree.query(pts[i], k=k+1)
        d = d[1:]; j = j[1:]
        order = np.argsort(d); d = d[order]; j = j[order]
        d[-1] = d[-1] if d[-1] != 0 else 1e-6
        ratios = (d[:-1] / d[-1]).tolist()
        desc.append(ratios); neigh.append([int(t) for t in j])
    return np.array(desc, float), neigh, pts

def candidate_catalog_for_image(D_img, D_cat, top_m=10):
    cands=[]
    for i in range(D_img.shape[0]):
        diff = D_cat - D_img[i]
        dist = np.sqrt(np.sum(diff*diff, axis=1))
        idx = np.argsort(dist)[:top_m]
        cands.append(idx.tolist())
    return cands

def estimate_f_from_pairs(pix_pts, cat_vecs):
    pairs = [(0,1),(0,2),(1,2)]
    vals=[]
    for a,b in pairs:
        xa,ya = pix_pts[a]; xb,yb = pix_pts[b]
        rij = math.hypot(xa-xb, ya-yb)
        dot = float(cat_vecs[a] @ cat_vecs[b]); dot = max(-1.0, min(1.0, dot))
        theta = math.degrees(math.acos(dot))
        t = math.tan(math.radians(theta)/2.0)
        if t <= 1e-9: continue
        vals.append(rij/(2.0*t))
    if not vals: return None
    return float(np.median(vals))

def kabsch_from_triplet(pix_pts, cat_vecs, cx, cy, f_pix, variant):
    rays = np.array([pixel_to_ray_stereo_f(x, y, cx, cy, f_pix) for (x,y) in pix_pts], float)
    A = (flip_matrix(variant) @ rays.T).T
    B = np.array(cat_vecs, float)
    return kabsch_cam_to_world(A, B)

def ransac_seed_fixed_f(stars, cand, catalog, cat_coords,
                        f, cx, cy,
                        trials=4000, min_inliers=4,
                        ang_gate_deg=0.45, px_gate=22.0, seed=7,
                        variants=("normal","flip_x","flip_y","flip_xy")):
    random.seed(seed); np.random.seed(seed)
    n = len(stars)
    if n < 3: return None
    w = np.array([s["brightness"] for s in stars], float)
    w = w/w.sum() if w.sum()>0 else np.ones(n)/n
    thr_chord = 2.0 * math.sin(math.radians(ang_gate_deg)/2.0)
    best = {"score":-1}
    idxs = list(range(n))
    tree = cKDTree(cat_coords)

    rays_all = np.array([pixel_to_ray_stereo_f(s["x"], s["y"], cx, cy, f) for s in stars], float)

    for _ in range(trials):
        i,j,k = list(np.random.choice(idxs, size=3, replace=False, p=w))
        if (np.hypot(stars[i]["x"]-stars[j]["x"], stars[i]["y"]-stars[j]["y"]) < 12 or
            np.hypot(stars[i]["x"]-stars[k]["x"], stars[i]["y"]-stars[k]["y"]) < 12 or
            np.hypot(stars[j]["x"]-stars[k]["x"], stars[j]["y"]-stars[k]["y"]) < 12):
            continue
        if not cand[i] or not cand[j] or not cand[k]: continue

        Ci = cand[i][:6]; Cj = cand[j][:6]; Ck = cand[k][:6]
        for ui in Ci:
            for uj in Cj:
                if uj == ui: continue
                for uk in Ck:
                    if uk == ui or uk == uj: continue

                    A = np.vstack([rays_all[i], rays_all[j], rays_all[k]])
                    B = np.vstack([catalog[ui]["vec"], catalog[uj]["vec"], catalog[uk]["vec"]])

                    for variant in variants:
                        F = flip_matrix(variant)
                        R = kabsch_cam_to_world((F @ A.T).T, B)
                        if not np.isfinite(R).all(): 
                            continue

                        world_dirs = (R @ (F @ rays_all.T)).T
                        d, idx = tree.query(world_dirs, k=1)
                        order = np.argsort(d)
                        used=set(); matches=[]
                        for ii in order:
                            if d[ii] > thr_chord: continue
                            jc = int(idx[ii])
                            if jc in used: continue
                            v_cam = F @ (R.T @ cat_coords[jc])
                            xp, yp, inside = ray_to_pixel_stereo_f(v_cam, cx, cy, f)
                            if not inside: continue
                            if math.hypot(xp - stars[ii]["x"], yp - stars[ii]["y"]) <= px_gate:
                                used.add(jc)
                                matches.append((ii, jc))

                        sc = len(matches)
                        if sc > best["score"]:
                            best = {"score":sc, "R":R, "f":f, "cx":cx, "cy":cy,
                                    "variant":variant, "matches":matches}
                            if sc >= max(min_inliers, int(0.6*n)):
                                return best
    return best if best["score"] > 0 else None

def refine_R(R_cw, f_pix, cx, cy, variant, stars, matches, cat_coords):
    if len(matches) < 3: return R_cw
    F = flip_matrix(variant)
    A=[]; B=[]
    for i,j in matches:
        ray = pixel_to_ray_stereo_f(stars[i]["x"], stars[i]["y"], cx, cy, f_pix)
        A.append((F @ ray))
        B.append(cat_coords[j])
    A=np.array(A,float); B=np.array(B,float)
    return kabsch_cam_to_world(A, B)

def project_catalog_pixels(R_cw, f_pix, cx, cy, variant, catalog, img_w, img_h):
    F = flip_matrix(variant)
    pts=[]
    for j,star in enumerate(catalog):
        v_cam = F @ (R_cw.T @ star["vec"])
        x,y,inside = ray_to_pixel_stereo_f(v_cam, cx, cy, f_pix)
        if not inside: continue
        if x<0 or x>=img_w or y<0 or y>=img_h: continue
        pts.append((j,x,y))
    return pts

def brightness_ranks(stars):
    order = np.argsort([-s["brightness"] for s in stars])
    rank = np.empty(len(stars), int); rank[order] = np.arange(len(stars))
    return rank

def mag_ranks(catalog, cat_idxs):
    mags = np.array([catalog[j]["vmag"] for j in cat_idxs], float)
    order = np.argsort(mags)
    rank = np.empty(len(cat_idxs), int); rank[order] = np.arange(len(cat_idxs))
    return rank

def pairwise_prune(matches, stars, catalog, R_cw, f_pix, cx, cy, variant, tol_deg=0.25, min_support=2):
    if len(matches) <= 2: return matches
    F = flip_matrix(variant)
    rays = [pixel_to_ray_stereo_f(stars[i]["x"], stars[i]["y"], cx, cy, f_pix) for i,_ in matches]
    raysF = [(F @ r) for r in rays]
    cats  = [catalog[j]["vec"] for _,j in matches]
    keep=[]
    for a,(i_a, j_a) in enumerate(matches):
        ok=0
        for b,(i_b, j_b) in enumerate(matches):
            if a==b: continue
            sep_img = angle_between(raysF[a], raysF[b])
            sep_cat = angle_between(cats[a], cats[b])
            if abs(sep_img - sep_cat) <= tol_deg:
                ok+=1
                if ok>=min_support: break
        if ok>=min_support:
            keep.append((i_a, j_a))
    return keep

def hungarian_assign_desc(R_cw, f_pix, cx, cy, variant,
                          stars, catalog, img_w, img_h,
                          cand_lists, px_gate=22.0, ang_gate_deg=0.45,
                          w_pos=1.0, w_brightness=0.12):
    proj = project_catalog_pixels(R_cw, f_pix, cx, cy, variant, catalog, img_w, img_h)
    if not proj: return []
    cat_idx = [t[0] for t in proj]
    cat_xy  = np.array([[t[1],t[2]] for t in proj], float)
    tree = cKDTree(cat_xy)
    det_xy = np.array([[s["x"], s["y"]] for s in stars], float)
    det_rank = brightness_ranks(stars)
    cat_rank = mag_ranks(catalog, cat_idx)
    F = flip_matrix(variant)
    thr_chord = 2.0 * math.sin(math.radians(ang_gate_deg)/2.0)

    N = len(stars); M = len(cat_idx)
    BIG = 1e6
    cost = np.full((N,M), BIG, float)

    for i,(x,y) in enumerate(det_xy):
        near = tree.query_ball_point([x,y], r=px_gate)
        if not near: continue
        cand_global = set(cand_lists[i])
        cand_local  = [k for k in near if cat_idx[k] in cand_global]
        if not cand_local:
            continue
        v_meas = F @ pixel_to_ray_stereo_f(x, y, cx, cy, f_pix)
        ray_world = R_cw @ v_meas
        ok_local=[]
        for k in cand_local:
            dot = float(ray_world @ catalog[cat_idx[k]]["vec"])
            if 1.0 - dot*dot < 0: dot = max(-1.0, min(1.0, dot))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            if ang <= ang_gate_deg*1.2:
                ok_local.append(k)
        if not ok_local: 
            ok_local = cand_local

        for k in ok_local:
            dx = x - cat_xy[k,0]; dy = y - cat_xy[k,1]
            d2 = dx*dx + dy*dy
            br = (det_rank[i] - cat_rank[k]) / max(1.0, max(N,M))
            c = w_pos * d2 + w_brightness * (br*br)
            cost[i,k] = c

    row_ind, col_ind = linear_sum_assignment(cost)
    matches=[]
    for i,jc in zip(row_ind, col_ind):
        if cost[i,jc] >= BIG*0.5:
            continue
        if math.hypot(det_xy[i,0]-cat_xy[jc,0], det_xy[i,1]-cat_xy[jc,1]) <= px_gate:
            matches.append((i, cat_idx[jc]))
    return matches

def annotate(image_path, stars, matches, catalog, out_path="annotated.png"):
    img=cv2.imread(image_path)
    for i,j in matches:
        x=int(round(stars[i]["x"])); y=int(round(stars[i]["y"]))
        cv2.circle(img,(x,y),6,(0,255,0),1)
        cv2.putText(img, catalog[j]["id"], (x+6,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path,img)
    print(f"[annotate] Saved {out_path}")

def intrinsics_from_fov(img_w, img_h, fov_deg, vertical=True):
    r_pix = (img_h/2.0) if vertical else (img_w/2.0)
    f = r_pix / (2.0 * math.tan(math.radians(fov_deg/4.0)))
    cx, cy = img_w/2.0, img_h/2.0
    return f, cx, cy

def angle_between_vecs_deg(u, v):
    un = np.asarray(u, float); vn = np.asarray(v, float)
    un /= (np.linalg.norm(un)+1e-12); vn /= (np.linalg.norm(vn)+1e-12)
    return math.degrees(math.acos(float(np.clip(un@vn, -1.0, 1.0))))

def quaternion_angular_distance(q1, q2):
    """
    Calculate angular distance between two quaternions in degrees
    Formula: angle = 2 * arccos(|q1 · q2|)
    """
    # Ensure quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate dot product
    dot_product = abs(np.dot(q1, q2))
    
    # Clamp to avoid numerical errors
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # Calculate angular distance
    angle_rad = 2 * np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def create_3d_reference_frame_plot(result, image_name, output_path):
    """
    Create 3D plot showing camera reference frame relative to inertial frame
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        print(f"[Warning] Plotly not available. Skipping 3D plot generation.")
        print(f"[Info] Install with: pip install plotly")
        return
    
    # Rotation matrix from QUEST
    R = result["rotation_matrix"]
    
    # Inertial frame (world) - standard XYZ
    inertial_x = np.array([1, 0, 0])
    inertial_y = np.array([0, 1, 0]) 
    inertial_z = np.array([0, 0, 1])
    
    # Camera frame rotated by R
    camera_x = R @ inertial_x
    camera_y = R @ inertial_y
    camera_z = R @ inertial_z
    
    fig = go.Figure()
    
    # Earth sphere at center (small blue sphere)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=12, color='lightblue', opacity=0.8),
        name='Earth',
        showlegend=True
    ))
    
    # Inertial frame (black, thin, clean)
    fig.add_trace(go.Scatter3d(
        x=[0, inertial_x[0]], y=[0, inertial_x[1]], z=[0, inertial_x[2]],
        mode='lines+text',
        line=dict(color='black', width=3),
        text=['', 'X'],
        textposition='middle center',
        textfont=dict(size=14, color='black'),
        name='Inertial X',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, inertial_y[0]], y=[0, inertial_y[1]], z=[0, inertial_y[2]],
        mode='lines+text',
        line=dict(color='black', width=3),
        text=['', 'Y'],
        textposition='middle center',
        textfont=dict(size=14, color='black'),
        name='Inertial Y',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, inertial_z[0]], y=[0, inertial_z[1]], z=[0, inertial_z[2]],
        mode='lines+text',
        line=dict(color='black', width=3),
        text=['', 'Z'],
        textposition='middle center',
        textfont=dict(size=14, color='black'),
        name='Inertial Z',
        showlegend=True
    ))
    
    # Camera frame (colored, dashed, slightly thinner)
    fig.add_trace(go.Scatter3d(
        x=[0, camera_x[0]], y=[0, camera_x[1]], z=[0, camera_x[2]],
        mode='lines+text',
        line=dict(color='red', width=4, dash='dash'),
        text=['', 'Xc'],
        textposition='middle center',
        textfont=dict(size=12, color='red'),
        name='Camera X',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, camera_y[0]], y=[0, camera_y[1]], z=[0, camera_y[2]],
        mode='lines+text',
        line=dict(color='green', width=4, dash='dash'),
        text=['', 'Yc'],
        textposition='middle center',
        textfont=dict(size=12, color='green'),
        name='Camera Y',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, camera_z[0]], y=[0, camera_z[1]], z=[0, camera_z[2]],
        mode='lines+text',
        line=dict(color='blue', width=4, dash='dash'),
        text=['', 'Zc'],
        textposition='middle center',
        textfont=dict(size=12, color='blue'),
        name='Camera Z',
        showlegend=True
    ))
    
    # Boresight direction vector (prominent but not too thick)
    boresight = result["boresight"]
    fig.add_trace(go.Scatter3d(
        x=[0, boresight[0]], y=[0, boresight[1]], z=[0, boresight[2]],
        mode='lines+markers+text',
        line=dict(color='purple', width=6),
        marker=dict(size=8, color='purple'),
        text=['', 'Boresight'],
        textposition='middle center',
        textfont=dict(size=14, color='purple'),
        name='Boresight Vector',
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text=f'3D Reference Frames - {image_name}<br>'
                 f'RA: {result["ra_deg"]:.3f}°, Dec: {result["dec_deg"]:.3f}°<br>'
                 f'Matches: {len(result["matches"])}, Error: {result["mean_error"]:.3f}°',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
            zaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
            bgcolor='white'
        ),
        width=900,
        height=700,
        font=dict(size=12),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"[3D Plot] Saved {output_path}")

def save_results_to_file(result1, result2, angular_distance, output_path):
    """
    Save detailed results to text file
    """
    import os
    from datetime import datetime
    
    with open(output_path, 'w', encoding='utf-8') as f:  # Fixed encoding
        f.write("="*80 + "\n")
        f.write("DUAL IMAGE ATTITUDE DETERMINATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: QUEST Algorithm\n\n")
        
        # Image 1 results
        f.write("IMAGE 1 RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(f"RA:  {result1['ra_deg']:.8f}°\n")
        f.write(f"Dec: {result1['dec_deg']:.8f}°\n")
        f.write(f"Quaternion (w,x,y,z): [{result1['quaternion'][0]:.8f}, {result1['quaternion'][1]:.8f}, {result1['quaternion'][2]:.8f}, {result1['quaternion'][3]:.8f}]\n")
        f.write(f"Euler Angles (roll,pitch,yaw): [{result1['euler_angles'][0]:.6f}°, {result1['euler_angles'][1]:.6f}°, {result1['euler_angles'][2]:.6f}°]\n")
        f.write(f"Matched stars: {len(result1['matches'])}\n")
        f.write(f"Star IDs: {', '.join(result1['matched_star_ids'])}\n")
        f.write(f"QUEST lambda_max: {result1['lambda_max']:.10f}\n")  # Fixed lambda character
        f.write(f"QUEST iterations: {result1['quest_iterations']}\n")
        f.write(f"Mean angular error: {result1['mean_error']:.6f}°\n")
        f.write(f"Max angular error: {result1['max_error']:.6f}°\n")
        f.write(f"RMS angular error: {result1['rms_error']:.6f}°\n")
        f.write(f"Min angular error: {result1['min_error']:.6f}°\n")
        
        f.write("\nRotation Matrix:\n")
        R1 = result1['rotation_matrix']
        for i in range(3):
            f.write(f"[{R1[i,0]:10.7f} {R1[i,1]:10.7f} {R1[i,2]:10.7f}]\n")
        
        f.write(f"\nBoresight vector: [{result1['boresight'][0]:.8f}, {result1['boresight'][1]:.8f}, {result1['boresight'][2]:.8f}]\n")
        
        f.write("\nIndividual star errors:\n")
        for star_id, error in zip(result1['matched_star_ids'], result1['angular_errors']):
            f.write(f"  {star_id:>8}: {error:7.4f}°\n")
        
        # Image 2 results
        f.write("\n\nIMAGE 2 RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(f"RA:  {result2['ra_deg']:.8f}°\n")
        f.write(f"Dec: {result2['dec_deg']:.8f}°\n")
        f.write(f"Quaternion (w,x,y,z): [{result2['quaternion'][0]:.8f}, {result2['quaternion'][1]:.8f}, {result2['quaternion'][2]:.8f}, {result2['quaternion'][3]:.8f}]\n")
        f.write(f"Euler Angles (roll,pitch,yaw): [{result2['euler_angles'][0]:.6f}°, {result2['euler_angles'][1]:.6f}°, {result2['euler_angles'][2]:.6f}°]\n")
        f.write(f"Matched stars: {len(result2['matches'])}\n")
        f.write(f"Star IDs: {', '.join(result2['matched_star_ids'])}\n")
        f.write(f"QUEST lambda_max: {result2['lambda_max']:.10f}\n")  # Fixed lambda character
        f.write(f"QUEST iterations: {result2['quest_iterations']}\n")
        f.write(f"Mean angular error: {result2['mean_error']:.6f}°\n")
        f.write(f"Max angular error: {result2['max_error']:.6f}°\n")
        f.write(f"RMS angular error: {result2['rms_error']:.6f}°\n")
        f.write(f"Min angular error: {result2['min_error']:.6f}°\n")
        
        f.write("\nRotation Matrix:\n")
        R2 = result2['rotation_matrix']
        for i in range(3):
            f.write(f"[{R2[i,0]:10.7f} {R2[i,1]:10.7f} {R2[i,2]:10.7f}]\n")
        
        f.write(f"\nBoresight vector: [{result2['boresight'][0]:.8f}, {result2['boresight'][1]:.8f}, {result2['boresight'][2]:.8f}]\n")
        
        f.write("\nIndividual star errors:\n")
        for star_id, error in zip(result2['matched_star_ids'], result2['angular_errors']):
            f.write(f"  {star_id:>8}: {error:7.4f}°\n")
        
        # Comparison results
        f.write("\n\nCOMPARISON ANALYSIS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Quaternion angular distance: {angular_distance:.8f}°\n")
        
        ra_diff = abs(result1['ra_deg'] - result2['ra_deg'])
        if ra_diff > 180:
            ra_diff = 360 - ra_diff
        dec_diff = abs(result1['dec_deg'] - result2['dec_deg'])
        f.write(f"RA difference: {ra_diff:.8f}°\n")
        f.write(f"Dec difference: {dec_diff:.8f}°\n")
        
        # Camera parameters
        f.write("\n\nCAMERA PARAMETERS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Focal length: {result1['f']:.2f} pixels\n")
        f.write(f"Principal point: ({result1['cx']:.1f}, {result1['cy']:.1f})\n")
        f.write(f"Image variant: {result1['variant']}\n")
    
    print(f"[Results File] Saved {output_path}")

def solve_two_images_with_quest(catalog_path,
                               image1_path,
                               image2_path,
                               **kwargs):
    """
    Solve attitude for two images and calculate angular distance between them
    """
    print(f"\n{'='*100}")
    print(f"DUAL IMAGE ATTITUDE DETERMINATION WITH QUEST")
    print(f"{'='*100}")
    
    # Solve first image
    print(f"\n🌟 PROCESSING FIRST IMAGE: {image1_path}")
    result1 = solve_single_image_with_quest(catalog_path, image1_path, **kwargs)
    
    if result1 is None:
        print(f"❌ Failed to solve first image")
        return None, None, None
    
    # Solve second image  
    print(f"\n🌟 PROCESSING SECOND IMAGE: {image2_path}")
    result2 = solve_single_image_with_quest(catalog_path, image2_path, **kwargs)
    
    if result2 is None:
        print(f"❌ Failed to solve second image")
        return result1, None, None
    
    # Calculate angular distance between quaternions
    q1 = result1["quaternion"]
    q2 = result2["quaternion"]
    angular_distance = quaternion_angular_distance(q1, q2)
    
    # Summary output (simplified)
    print(f"\n{'='*100}")
    print(f"DUAL IMAGE COMPARISON RESULTS")
    print(f"{'='*100}")
    
    print(f"\nIMAGE 1 ({image1_path}):")
    print(f"  RA:  {result1['ra_deg']:.6f}°")
    print(f"  Dec: {result1['dec_deg']:.6f}°")
    print(f"  Quaternion: [{result1['quaternion'][0]:.6f}, {result1['quaternion'][1]:.6f}, {result1['quaternion'][2]:.6f}, {result1['quaternion'][3]:.6f}]")
    print(f"  Matched stars: {len(result1['matches'])}")
    print(f"  Mean error: {result1['mean_error']:.4f}°")
    
    print(f"\nIMAGE 2 ({image2_path}):")
    print(f"  RA:  {result2['ra_deg']:.6f}°")
    print(f"  Dec: {result2['dec_deg']:.6f}°")
    print(f"  Quaternion: [{result2['quaternion'][0]:.6f}, {result2['quaternion'][1]:.6f}, {result2['quaternion'][2]:.6f}, {result2['quaternion'][3]:.6f}]")
    print(f"  Matched stars: {len(result2['matches'])}")
    print(f"  Mean error: {result2['mean_error']:.4f}°")
    
    print(f"\nANGULAR DISTANCE:")
    print(f"  Quaternion angular distance: {angular_distance:.6f}°")
    
    # Generate outputs
    import os
    base_dir = os.path.dirname(image1_path) if os.path.dirname(image1_path) else "."
    
    # Save detailed results to file
    results_file = os.path.join(base_dir, "attitude_results.txt")
    save_results_to_file(result1, result2, angular_distance, results_file)
    
    # Create 3D plots (requires plotly)
    plot1_path = os.path.join(base_dir, "reference_frame_image1.html")
    plot2_path = os.path.join(base_dir, "reference_frame_image2.html")
    
    print(f"\n[Generating 3D plots...]")
    create_3d_reference_frame_plot(result1, "Image 1", plot1_path)
    create_3d_reference_frame_plot(result2, "Image 2", plot2_path)
    
    return result1, result2, angular_distance

def solve_single_image_with_quest(catalog_path,
                                 image_path,
                                 *,
                                 fov_deg=66.0,
                                 fov_is_vertical=True,
                                 mag_limit=3,
                                 max_image_stars=26,
                                 k_desc=5,
                                 top_m=10,
                                 ang_gate_deg=0.45,
                                 px_gate_seed=22.0,
                                 px_gate_assign=20.0,
                                 refine_loops=1,  # mantenuto per compatibilità (non usato)
                                 seed=7,
                                 annotate_suffix="_quest.png"):
    """
    Star identification + QUEST attitude determination
    Pipeline: Detection → RANSAC matching → QUEST final attitude
    """
    random.seed(seed); np.random.seed(seed)

    print(f"\n{'='*80}")
    print(f"STAR IDENTIFICATION AND ATTITUDE DETERMINATION WITH QUEST")
    print(f"{'='*80}")
    
    # Detection
    print(f"\n[STEP 1] Detecting stars in {image_path}")
    stars_all, W, H = detect_stars(image_path, threshold_rel=0.60, min_area=2, dedupe_px=2.0)
    stars = stars_all[:max_image_stars]
    print(f"[INFO] Image size: {W} x {H} pixels")
    print(f"[INFO] Using {len(stars)} brightest stars (from {len(stars_all)} detected)")
    
    if len(stars) < 6:
        print(f"[ERROR] Too few detections in {image_path}")
        return None

    # Fixed intrinsics from FOV
    print(f"\n[STEP 2] Camera model setup")
    f, cx, cy = intrinsics_from_fov(W, H, fov_deg, vertical=fov_is_vertical)
    fov_type = "vertical" if fov_is_vertical else "horizontal"
    print(f"[INFO] FOV: {fov_deg}° ({fov_type})")
    print(f"[INFO] Focal length: {f:.1f} pixels")
    print(f"[INFO] Principal point: ({cx:.1f}, {cy:.1f})")

    # Catalog and descriptors
    print(f"\n[STEP 3] Loading catalog and computing descriptors")
    catalog = parse_catalog(catalog_path, mag_limit=mag_limit)
    D_img, _, _ = build_image_descriptors(stars, k=k_desc)
    D_cat, _, ccoords = build_catalog_descriptors(catalog, k=k_desc, neighbor_max_deg=70.0)
    cand = candidate_catalog_for_image(D_img, D_cat, top_m=top_m)
    print(f"[INFO] Using {len(catalog)} catalog stars with V ≤ {mag_limit}")

    # Seed pose
    print(f"\n[STEP 4] Initial pose estimation via RANSAC")
    best = ransac_seed_fixed_f(stars, cand, catalog, ccoords,
                               f, cx, cy,
                               trials=4000, min_inliers=4,
                               ang_gate_deg=ang_gate_deg, px_gate=px_gate_seed, seed=seed)
    if not best or best["score"] < 3:
        print(f"[ERROR] No initial pose found for {image_path}")
        return None
    print(f"[INFO] Initial seed found {best['score']} matches")

    # Star matching (single pass)
    print(f"\n[STEP 5] Final star matching")
    matches = hungarian_assign_desc(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
                                    stars, catalog, W, H,
                                    cand, px_gate=px_gate_assign, ang_gate_deg=ang_gate_deg,
                                    w_pos=1.0, w_brightness=0.12)
    matches = pairwise_prune(matches, stars, catalog, best["R"], best["f"], best["cx"], best["cy"], best["variant"],
                             tol_deg=0.25, min_support=2)
    print(f"[INFO] Found {len(matches)} robust star matches")

    if len(matches) < 3:
        print(f"[ERROR] Insufficient final matches: {len(matches)}")
        return None

    # QUEST attitude determination (final solution)
    print(f"\n[STEP 6] QUEST attitude determination (final solution)")
    print(f"[INFO] Computing optimal attitude from {len(matches)} star matches")
    
    # Prepare vectors for QUEST
    F = flip_matrix(best["variant"])
    body_vectors = []
    inertial_vectors = []
    matched_star_ids = []
    
    for i, j in matches:
        # Body vector (camera ray)
        ray = pixel_to_ray_stereo_f(stars[i]["x"], stars[i]["y"], cx, cy, f)
        body_vec = F @ ray
        body_vectors.append(body_vec)
        
        # Inertial vector (catalog star direction)
        inertial_vectors.append(catalog[j]["vec"])
        matched_star_ids.append(catalog[j]["id"])

    body_vectors = np.array(body_vectors)
    inertial_vectors = np.array(inertial_vectors)
    
    # Brightness-based weights
    matched_brightnesses = [stars[i]["brightness"] for i, j in matches]
    weights = np.array(matched_brightnesses)
    weights = weights / np.sum(weights)

    # Run QUEST
    q_quest, lambda_max, quest_iterations = QUEST.quest_method(
        body_vectors, inertial_vectors, weights=weights, tol=1e-12, max_iter=50)
    
    # Convert to rotation matrix
    R_quest = quaternion_to_rotation_matrix(q_quest)
    
    # Calculate errors
    residuals = []
    for i, (body_vec, inertial_vec) in enumerate(zip(body_vectors, inertial_vectors)):
        predicted = R_quest @ body_vec
        error_rad = np.arccos(np.clip(np.dot(predicted, inertial_vec), -1, 1))
        residuals.append(np.degrees(error_rad))

    # Boresight calculation
    boresight_body = np.array([0, 0, 1.0])  # Camera Z-axis
    boresight_world = R_quest @ (F @ boresight_body)
    boresight_world /= (np.linalg.norm(boresight_world) + 1e-12)
    
    bx, by, bz = boresight_world
    dec = math.degrees(math.asin(np.clip(bz, -1, 1)))
    ra = (math.degrees(math.atan2(by, bx)) % 360.0)
    
    # Euler angles
    roll, pitch, yaw = rotation_matrix_to_euler(R_quest)

    # Results summary
    print(f"\n{'='*80}")
    print(f"FINAL ATTITUDE DETERMINATION RESULTS")
    print(f"{'='*80}")
    print(f"Method: QUEST algorithm (optimal solution)")
    print(f"Stars matched: {len(matches)}/{len(stars)} used stars")
    print(f"Matched star IDs: {', '.join(matched_star_ids[:10])}{'...' if len(matched_star_ids) > 10 else ''}")
    print(f"\nBORESIGHT (Image Center):")
    print(f"  RA:  {ra:.6f}°")
    print(f"  Dec: {dec:.6f}°")
    print(f"\nQUATERNION (w, x, y, z):")
    print(f"  [{q_quest[0]:.8f}, {q_quest[1]:.8f}, {q_quest[2]:.8f}, {q_quest[3]:.8f}]")
    print(f"\nEULER ANGLES:")
    print(f"  Roll:  {roll:.6f}°")
    print(f"  Pitch: {pitch:.6f}°")
    print(f"  Yaw:   {yaw:.6f}°")
    print(f"\nROTATION MATRIX:")
    for i in range(3):
        print(f"  [{R_quest[i,0]:9.6f} {R_quest[i,1]:9.6f} {R_quest[i,2]:9.6f}]")
    print(f"\nQUEST SOLUTION QUALITY:")
    print(f"  λ_max: {lambda_max:.10f}")
    print(f"  Newton-Raphson iterations: {quest_iterations}")
    print(f"  Mean angular error: {np.mean(residuals):.4f}°")
    print(f"  Max angular error:  {np.max(residuals):.4f}°")
    print(f"  RMS angular error:  {np.sqrt(np.mean(np.array(residuals)**2)):.4f}°")
    print(f"  Min angular error:  {np.min(residuals):.4f}°")

    # Individual star errors
    print(f"\nINDIVIDUAL STAR MATCHING ERRORS:")
    for i, (star_id, error) in enumerate(zip(matched_star_ids, residuals)):
        brightness = matched_brightnesses[i]
        print(f"  {star_id:>8}: {error:7.3f}° (brightness: {brightness:8.1f})")

    # Annotate image
    import os
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base}{annotate_suffix}"
    annotate(image_path, stars, matches, catalog, out_path=output_path)

    return {
        "method": "QUEST",
        "boresight": boresight_world,
        "ra_deg": ra,
        "dec_deg": dec,
        "quaternion": q_quest,
        "rotation_matrix": R_quest,
        "euler_angles": (roll, pitch, yaw),
        "matches": matches,
        "matched_star_ids": matched_star_ids,
        "stars": stars,
        "f": f, "cx": cx, "cy": cy,
        "variant": best["variant"],
        "lambda_max": lambda_max,
        "quest_iterations": quest_iterations,
        "angular_errors": residuals,
        "mean_error": np.mean(residuals),
        "max_error": np.max(residuals),
        "rms_error": np.sqrt(np.mean(np.array(residuals)**2)),
        "min_error": np.min(residuals)
    }

# Usage example
if __name__ == "__main__":
    catalog_txt = r"c:\Users\157205\Desktop\SPARCS-main\HipparcosCatalog.txt"
    image1_png = r"c:\Users\157205\Downloads\pov2.png"  # First image
    image2_png = r"c:\Users\157205\Downloads\pov1.png"  # Second image
    
    # Solve both images and calculate angular distance
    result1, result2, angular_distance = solve_two_images_with_quest(
        catalog_txt, image1_png, image2_png,
        fov_deg=66.0,
        fov_is_vertical=True,
        mag_limit=3,
        max_image_stars=26,
        k_desc=5,
        top_m=10,
        ang_gate_deg=0.45,
        px_gate_seed=22.0,
        px_gate_assign=20.0,
        refine_loops=1,
        seed=7,
        annotate_suffix="_quest.png"
    )
    
    # Additional summary
    if result1 and result2:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   Quaternion angular distance: {angular_distance:.6f}°")
        print(f"   📄 Results saved to: attitude_results.txt")
        print(f"   📊 3D plots: reference_frame_image1.html, reference_frame_image2.html")
    
    elif result1:
        print(f"\n⚠️  Only first image solved successfully")
    else:
        print(f"\n❌ Both images failed to solve")