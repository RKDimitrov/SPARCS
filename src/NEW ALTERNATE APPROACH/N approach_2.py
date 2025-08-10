import csv
import math
import random
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# ---------------- RA/Dec, catalog (.txt) ----------------

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

# ---------------- star detection ----------------

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

# ---------------- stereographic camera (learn f, cx, cy) ----------------

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

# ---------------- geometry ----------------

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

# ---------------- ratio descriptors (stereo-invariant) ----------------

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

# ---------------- seed (learn f,cx,cy,R) via triplets ----------------

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

def ransac_seed(stars, cand, catalog, cat_coords,
                trials=9000, ransac_min_inliers=7,
                ang_gate_deg=0.45, px_gate=22.0, seed=7):
    random.seed(seed)
    n = len(stars)
    brightness = np.array([s["brightness"] for s in stars], float)
    w = brightness/brightness.sum() if brightness.sum()>0 else np.ones(n)/n
    best={"score":-1}
    idxs = list(range(n))
    thr = 2.0 * math.sin(math.radians(ang_gate_deg)/2.0)

    for _ in range(trials):
        tri = list(np.random.choice(idxs, size=3, replace=False, p=w))
        i,j,k = tri
        if (np.hypot(stars[i]["x"]-stars[j]["x"], stars[i]["y"]-stars[j]["y"]) < 12 or
            np.hypot(stars[i]["x"]-stars[k]["x"], stars[i]["y"]-stars[k]["y"]) < 12 or
            np.hypot(stars[j]["x"]-stars[k]["x"], stars[j]["y"]-stars[k]["y"]) < 12):
            continue
        try:
            ui = random.choice(cand[i][:4]); uj = random.choice(cand[j][:4]); uk = random.choice(cand[k][:4])
        except Exception:
            continue
        if len({ui,uj,uk})<3: continue

        pix_pts = [(stars[i]["x"], stars[i]["y"]),
                   (stars[j]["x"], stars[j]["y"]),
                   (stars[k]["x"], stars[k]["y"])]
        cat_vecs = [catalog[ui]["vec"], catalog[uj]["vec"], catalog[uk]["vec"]]
        cx = (pix_pts[0][0]+pix_pts[1][0]+pix_pts[2][0])/3.0
        cy = (pix_pts[0][1]+pix_pts[1][1]+pix_pts[2][1])/3.0
        f_pix = estimate_f_from_pairs(pix_pts, cat_vecs)
        if f_pix is None or not np.isfinite(f_pix) or f_pix <= 10: continue

        for variant in ("normal","flip_x","flip_y","flip_xy"):
            R = kabsch_from_triplet(pix_pts, cat_vecs, cx, cy, f_pix, variant)
            if R is None or not np.isfinite(R).all(): continue

            # quick seed associations (angular + pixel)
            img_rays = np.array([pixel_to_ray_stereo_f(s["x"], s["y"], cx, cy, f_pix) for s in stars], float)
            F = flip_matrix(variant)
            world_dirs=(R @ (F @ img_rays.T)).T
            tree=cKDTree(cat_coords)
            d,idx = tree.query(world_dirs, k=1)
            order=np.argsort(d)
            used=set(); matches=[]
            for ii in order:
                if d[ii] > thr: continue
                jcat=int(idx[ii])
                if jcat in used: continue
                v_cam = F @ (R.T @ cat_coords[jcat])
                xp, yp, inside = ray_to_pixel_stereo_f(v_cam, cx, cy, f_pix)
                if not inside: continue
                pe = math.hypot(xp - stars[ii]["x"], yp - stars[ii]["y"])
                if pe <= px_gate:
                    used.add(jcat)
                    matches.append((ii, jcat))
            sc=len(matches)
            if sc>best["score"]:
                best={"score":sc, "R":R, "f":f_pix, "cx":cx, "cy":cy, "variant":variant, "matches":matches}
                if sc>=max(ransac_min_inliers, int(0.6*n)): return best
    return best

# ---------------- refinement tools ----------------

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

def refine_intrinsics_grid(R_cw, f_pix, cx, cy, variant, stars, matches, cat_coords):
    if len(matches) < 3: return f_pix, cx, cy
    F = flip_matrix(variant)
    def sse(f2, cx2, cy2):
        s=0.0
        for i,j in matches:
            v_cam = F @ (R_cw.T @ cat_coords[j])
            xp,yp,_ = ray_to_pixel_stereo_f(v_cam, cx2, cy2, f2)
            dx = xp - stars[i]["x"]; dy = yp - stars[i]["y"]
            s += dx*dx + dy*dy
        return s
    best=(1e18,f_pix,cx,cy)
    for sf in (0.985,1.0,1.015):
        for dx in (-6.0,0.0,6.0):
            for dy in (-6.0,0.0,6.0):
                f2, cx2, cy2 = f_pix*sf, cx+dx, cy+dy
                val = sse(f2, cx2, cy2)
                if val < best[0]: best=(val,f2,cx2,cy2)
    return best[1], best[2], best[3]

def project_catalog_pixels(R_cw, f_pix, cx, cy, variant, catalog, img_w, img_h):
    F = flip_matrix(variant)
    pts=[]
    for j,star in enumerate(catalog):
        v_cam = F @ (R_cw.T @ star["vec"])
        x,y,inside = ray_to_pixel_stereo_f(v_cam, cx, cy, f_pix)
        if not inside: continue
        if x<0 or x>=img_w or y<0 or y>=img_h: continue
        pts.append((j,x,y))
    return pts  # (cat_idx, x, y)

def brightness_ranks(stars):
    order = np.argsort([-s["brightness"] for s in stars])
    rank = np.empty(len(stars), int); rank[order] = np.arange(len(stars))
    return rank

def mag_ranks(catalog, cat_idxs):
    mags = np.array([catalog[j]["vmag"] for j in cat_idxs], float)
    order = np.argsort(mags)  # smaller mag = brighter
    rank = np.empty(len(cat_idxs), int); rank[order] = np.arange(len(cat_idxs))
    return rank

# ---------------- pairwise-consistency prune ----------------

def pairwise_prune(matches, stars, catalog, R_cw, f_pix, cx, cy, variant, tol_deg=0.25, min_support=2):
    if len(matches) <= 2: return matches
    F = flip_matrix(variant)
    # Build camera-frame rays for all detections
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

# ---------------- descriptor-constrained Hungarian assignment ----------------

def hungarian_assign_desc(R_cw, f_pix, cx, cy, variant,
                          stars, catalog, img_w, img_h,
                          cand_lists, px_gate=22.0, ang_gate_deg=0.45,
                          w_pos=1.0, w_brightness=0.12):
    # project catalog to pixels
    proj = project_catalog_pixels(R_cw, f_pix, cx, cy, variant, catalog, img_w, img_h)
    if not proj: return []
    cat_idx = [t[0] for t in proj]
    cat_xy  = np.array([[t[1],t[2]] for t in proj], float)
    # KD-tree in pixel space
    tree = cKDTree(cat_xy)
    # per-detector pixel coord
    det_xy = np.array([[s["x"], s["y"]] for s in stars], float)
    det_rank = brightness_ranks(stars)
    cat_rank = mag_ranks(catalog, cat_idx)
    # angular gate (per detection)
    F = flip_matrix(variant)
    thr_chord = 2.0 * math.sin(math.radians(ang_gate_deg)/2.0)

    N = len(stars); M = len(cat_idx)
    BIG = 1e6
    cost = np.full((N,M), BIG, float)

    for i,(x,y) in enumerate(det_xy):
        # candidate set = (pixel gate) ∩ (descriptor top list mapped into projected set)
        near = tree.query_ball_point([x,y], r=px_gate)
        if not near: continue
        cand_global = set(cand_lists[i])              # global catalog indices by descriptor
        cand_local  = [k for k in near if cat_idx[k] in cand_global]
        if not cand_local:
            continue
        # angular gate around predicted world ray (extra safety)
        v_meas = F @ pixel_to_ray_stereo_f(x, y, cx, cy, f_pix)
        ray_world = R_cw @ v_meas
        # precompute angular separations
        # build a small list that passes the angular gate too
        ok_local=[]
        for k in cand_local:
            dot = float(ray_world @ catalog[cat_idx[k]]["vec"])
            if 1.0 - dot*dot < 0: dot = max(-1.0, min(1.0, dot))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            if ang <= ang_gate_deg*1.2:  # a tad looser than global ang gate
                ok_local.append(k)
        if not ok_local: 
            ok_local = cand_local  # fallback to pixel-only if too strict

        for k in ok_local:
            dx = x - cat_xy[k,0]; dy = y - cat_xy[k,1]
            d2 = dx*dx + dy*dy
            br = (det_rank[i] - cat_rank[k]) / max(1.0, max(N,M))
            c = w_pos * d2 + w_brightness * (br*br)
            cost[i,k] = c

    row_ind, col_ind = linear_sum_assignment(cost)
    matches=[]
    for i,jc in zip(row_ind, col_ind):
        if cost[i,jc] >= BIG*0.5:  # infeasible
            continue
        # final tight pixel gate again
        if math.hypot(det_xy[i,0]-cat_xy[jc,0], det_xy[i,1]-cat_xy[jc,1]) <= px_gate:
            matches.append((i, cat_idx[jc]))
    return matches

# ---------------- annotate & debug ----------------

def debug_dump(params, stars, catalog):
    R = params["R"]; f_pix=params["f"]; cx=params["cx"]; cy=params["cy"]; variant=params["variant"]
    print(f"[debug] image params: f={f_pix:.1f}px, cx={cx:.1f}, cy={cy:.1f}, variant={variant}")
    print("[debug] sample matches:")
    F = flip_matrix(variant)
    for (ii, jj) in params["matches"][:12]:
        xd, yd = stars[ii]["x"], stars[ii]["y"]
        v_cam = F @ (R.T @ catalog[jj]["vec"])
        xp, yp, inside = ray_to_pixel_stereo_f(v_cam, cx, cy, f_pix)
        pe = math.hypot((xp or 0)-xd, (yp or 0)-yd) if inside else float("inf")
        v_meas = F @ pixel_to_ray_stereo_f(xd, yd, cx, cy, f_pix)
        ray_world = R @ v_meas
        dotc = max(-1.0, min(1.0, float(ray_world @ catalog[jj]["vec"])))
        ang_err = math.degrees(math.acos(dotc))*60.0
        print(f"  det=({xd:.1f},{yd:.1f}) -> {catalog[jj]['id']}  px_err={pe:.1f}  ang_err={ang_err:.1f}'")

def annotate(image_path, stars, matches, catalog, out_path="annotated.png"):
    img=cv2.imread(image_path)
    for i,j in matches:
        x=int(round(stars[i]["x"])); y=int(round(stars[i]["y"]))
        cv2.circle(img,(x,y),6,(0,255,0),1)
        cv2.putText(img, catalog[j]["id"], (x+6,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path,img)
    print(f"[annotate] Saved {out_path}")

# # ---------------- main ----------------

# def main(catalog_path,
#          image_path,
#          mag_limit=3,
#          max_image_stars=26,
#          k_desc=5,
#          top_m=10,
#          # gates
#          ang_gate_deg=0.45,
#          px_gate_seed=22.0,
#          px_gate_assign_1=24.0,   # pass-1 (loose)
#          px_gate_assign_2=16.0,   # pass-2 (tighten)
#          # RANSAC
#          ransac_trials=9000,
#          ransac_min_inliers=7,
#          refine_loops=2,
#          seed=7):

#     random.seed(seed); np.random.seed(seed)

#     # detections
#     stars_all, img_w, img_h = detect_stars(image_path, threshold_rel=0.62, min_area=2, dedupe_px=2.0)
#     stars = stars_all[:max_image_stars]
#     print(f"[debug] image size = {img_w} x {img_h}px; using {len(stars)} detections")

#     # catalog + descriptors
#     catalog = parse_catalog(catalog_path, mag_limit=mag_limit)
#     D_img, _, _ = build_image_descriptors(stars, k=k_desc)
#     D_cat, _, cat_coords = build_catalog_descriptors(catalog, k=k_desc, neighbor_max_deg=70.0)
#     cand = candidate_catalog_for_image(D_img, D_cat, top_m=top_m)   # list per detection (global indices)

#     # seed pose
#     best = ransac_seed(stars, cand, catalog, cat_coords,
#                        trials=ransac_trials, ransac_min_inliers=ransac_min_inliers,
#                        ang_gate_deg=ang_gate_deg, px_gate=px_gate_seed, seed=seed)
#     if best.get("score",-1) < 3:
#         print("[final] No seed pose. Try: px_gate_seed=28, max_image_stars=22, threshold_rel=0.66.")
#         return

#     # -------- PASS 1: descriptor-constrained Hungarian, pairwise-prune, refine --------
#     matches = hungarian_assign_desc(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
#                                     stars, catalog, img_w, img_h,
#                                     cand, px_gate=px_gate_assign_1, ang_gate_deg=ang_gate_deg,
#                                     w_pos=1.0, w_brightness=0.12)
#     matches = pairwise_prune(matches, stars, catalog, best["R"], best["f"], best["cx"], best["cy"], best["variant"],
#                              tol_deg=0.25, min_support=2)

#     for _ in range(refine_loops):
#         best["R"] = refine_R(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
#                              stars, matches, cat_coords)
#         best["f"], best["cx"], best["cy"] = refine_intrinsics_grid(best["R"], best["f"], best["cx"], best["cy"],
#                                                                    best["variant"], stars, matches, cat_coords)
#         matches = hungarian_assign_desc(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
#                                         stars, catalog, img_w, img_h,
#                                         cand, px_gate=px_gate_assign_1, ang_gate_deg=ang_gate_deg,
#                                         w_pos=1.0, w_brightness=0.12)
#         matches = pairwise_prune(matches, stars, catalog, best["R"], best["f"], best["cx"], best["cy"], best["variant"],
#                                  tol_deg=0.25, min_support=2)

#     # -------- PASS 2: tighten gates and finalize --------
#     matches = hungarian_assign_desc(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
#                                     stars, catalog, img_w, img_h,
#                                     cand, px_gate=px_gate_assign_2, ang_gate_deg=0.35,
#                                     w_pos=1.0, w_brightness=0.16)
#     matches = pairwise_prune(matches, stars, catalog, best["R"], best["f"], best["cx"], best["cy"], best["variant"],
#                              tol_deg=0.22, min_support=2)
#     best["matches"] = matches

#     # attitude
#     b_world = best["R"] @ np.array([0,0,1.0]); b_world/= (np.linalg.norm(b_world)+1e-12)
#     bx,by,bz = b_world
#     dec = math.degrees(math.asin(bz))
#     ra  = (math.degrees(math.atan2(by,bx)) % 360.0)
#     print(f"[best] inliers={len(best['matches'])}/{len(stars)}  (stereo; learned f & center; desc+Hungarian+pairwise)")
#     print(f"[attitude] Boresight ≈ RA {ra:.2f}°, Dec {dec:.2f}°")
#     debug_dump(best, stars, catalog)
#     annotate(image_path, stars, best["matches"], catalog, out_path="annotated.png")

# # -------- entry --------
# if __name__ == "__main__":
#     catalog_txt = r"C:\Users\kiira\OneDrive\Desktop\Space Challenges\N_Approach\hipparcos_N3.txt"
#     image_png   = r"C:\Users\kiira\OneDrive\Desktop\Space Challenges\N_Approach\pov.png"
#     main(
#         catalog_path=catalog_txt,
#         image_path=image_png,
#         mag_limit=3,
#         max_image_stars=26,      # you can try 28–30 once IDs look clean
#         k_desc=5,
#         top_m=10,
#         ang_gate_deg=0.45,
#         px_gate_seed=22.0,
#         px_gate_assign_1=24.0,   # pass-1
#         px_gate_assign_2=16.0,   # pass-2 (tight)
#         ransac_trials=9000,
#         ransac_min_inliers=7,
#         refine_loops=2,
#         seed=7
#     )




# ============================ Fixed-FOV attitude (center = boresight) ============================

def intrinsics_from_fov(img_w, img_h, fov_deg, vertical=True):
    """
    Stereographic: r = 2 f tan(theta/2). At the top/bottom (or left/right) edge,
    r_pix = (H/2) or (W/2) and theta = FOV/2.
    => f = r_pix / (2 * tan((FOV/2)/2)) = r_pix / (2 * tan(FOV/4)).
    """
    r_pix = (img_h/2.0) if vertical else (img_w/2.0)
    f = r_pix / (2.0 * math.tan(math.radians(fov_deg/4.0)))
    cx, cy = img_w/2.0, img_h/2.0
    return f, cx, cy

def ransac_seed_fixed_f(stars, cand, catalog, cat_coords,
                        f, cx, cy,
                        trials=4000, min_inliers=4,
                        ang_gate_deg=0.45, px_gate=22.0, seed=7,
                        variants=("normal","flip_x","flip_y","flip_xy")):
    """
    RANSAC that estimates ONLY rotation (R). f,cx,cy are fixed to the image center & given FOV.
    """
    random.seed(seed); np.random.seed(seed)
    n = len(stars)
    if n < 3: return None
    w = np.array([s["brightness"] for s in stars], float)
    w = w/w.sum() if w.sum()>0 else np.ones(n)/n
    thr_chord = 2.0 * math.sin(math.radians(ang_gate_deg)/2.0)
    best = {"score":-1}
    idxs = list(range(n))
    tree = cKDTree(cat_coords)

    # Precompute rays once for speed
    rays_all = np.array([pixel_to_ray_stereo_f(s["x"], s["y"], cx, cy, f) for s in stars], float)

    for _ in range(trials):
        i,j,k = list(np.random.choice(idxs, size=3, replace=False, p=w))
        # avoid tiny triangles in pixels
        if (np.hypot(stars[i]["x"]-stars[j]["x"], stars[i]["y"]-stars[j]["y"]) < 12 or
            np.hypot(stars[i]["x"]-stars[k]["x"], stars[i]["y"]-stars[k]["y"]) < 12 or
            np.hypot(stars[j]["x"]-stars[k]["x"], stars[j]["y"]-stars[k]["y"]) < 12):
            continue
        if not cand[i] or not cand[j] or not cand[k]: continue

        # try a few top descriptor candidates
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

                        # quick associate all detections with fixed f,cx,cy
                        world_dirs = (R @ (F @ rays_all.T)).T
                        d, idx = tree.query(world_dirs, k=1)
                        order = np.argsort(d)
                        used=set(); matches=[]
                        for ii in order:
                            if d[ii] > thr_chord: continue
                            jc = int(idx[ii])
                            if jc in used: continue
                            # pixel reprojection check
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

def solve_one_image_fixed_fov(catalog_path, image_path, fov_deg,
                              fov_is_vertical=True,
                              # detection / pipeline knobs kept minimal & fast
                              mag_limit=3, max_image_stars=26,
                              k_desc=5, top_m=10,
                              ang_gate_deg=0.45,
                              px_gate_seed=22.0,
                              px_gate_assign=20.0,
                              refine_loops=1,
                              seed=7,
                              annotate_suffix="_fixedFOV.png"):
    """
    Full solve for a single image with (f, cx, cy) locked to image center and given FOV.
    Returns dict with R, boresight, RA/Dec and matches.
    """
    random.seed(seed); np.random.seed(seed)

    # detect
    stars_all, W, H = detect_stars(image_path, threshold_rel=0.60, min_area=2, dedupe_px=2.0)
    stars = stars_all[:max_image_stars]
    if len(stars) < 6:
        print(f"[fixedFOV] Too few detections in {image_path}."); 
        return None

    # fixed intrinsics from FOV and center
    f, cx, cy = intrinsics_from_fov(W, H, fov_deg, vertical=fov_is_vertical)

    # bright catalog + descriptors
    catalog = parse_catalog(catalog_path, mag_limit=mag_limit)
    D_img, _, _   = build_image_descriptors(stars, k=k_desc)
    D_cat, _, ccoords = build_catalog_descriptors(catalog, k=k_desc, neighbor_max_deg=70.0)
    cand = candidate_catalog_for_image(D_img, D_cat, top_m=top_m)

    # seed R only
    best = ransac_seed_fixed_f(stars, cand, catalog, ccoords,
                               f, cx, cy,
                               trials=4000, min_inliers=4,
                               ang_gate_deg=ang_gate_deg, px_gate=px_gate_seed, seed=seed)
    if not best or best["score"] < 3:
        print(f"[fixedFOV] No seed pose for {image_path}.")
        return None

    # small assignment (descriptor-constrained, single pass) + prune + refine R only
    matches = hungarian_assign_desc(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
                                    stars, catalog, W, H,
                                    cand, px_gate=px_gate_assign, ang_gate_deg=ang_gate_deg,
                                    w_pos=1.0, w_brightness=0.12)
    matches = pairwise_prune(matches, stars, catalog, best["R"], best["f"], best["cx"], best["cy"], best["variant"],
                             tol_deg=0.25, min_support=2)

    for _ in range(refine_loops):
        best["R"] = refine_R(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
                             stars, matches, ccoords)
        matches = hungarian_assign_desc(best["R"], best["f"], best["cx"], best["cy"], best["variant"],
                                        stars, catalog, W, H,
                                        cand, px_gate=px_gate_assign, ang_gate_deg=ang_gate_deg,
                                        w_pos=1.0, w_brightness=0.12)
        matches = pairwise_prune(matches, stars, catalog, best["R"], best["f"], best["cx"], best["cy"], best["variant"],
                                 tol_deg=0.23, min_support=2)

    # boresight of image CENTER
    b_world = best["R"] @ np.array([0,0,1.0])
    b_world /= (np.linalg.norm(b_world) + 1e-12)
    bx,by,bz = b_world
    dec = math.degrees(math.asin(bz))
    ra  = (math.degrees(math.atan2(by,bx)) % 360.0)

    print(f"[fixedFOV] {image_path}: matches={len(matches)}/{len(stars)}  RA={ra:.3f}°, Dec={dec:.3f}°  (boresight=center)")
    # annotate
    import os
    base = os.path.splitext(os.path.basename(image_path))[0]
    annotate(image_path, stars, matches, catalog, out_path=f"{base}{annotate_suffix}")

    return {"R":best["R"], "boresight":b_world, "ra_deg":ra, "dec_deg":dec,
            "matches":matches, "stars":stars, "f":f, "cx":cx, "cy":cy, "variant":best["variant"]}

def angle_between_vecs_deg(u, v):
    un = np.asarray(u, float); vn = np.asarray(v, float)
    un /= (np.linalg.norm(un)+1e-12); vn /= (np.linalg.norm(vn)+1e-12)
    return math.degrees(math.acos(float(np.clip(un@vn, -1.0, 1.0))))

def solve_two_images_fixed_fov(catalog_path, image1_path, image2_path,
                               fov_deg=66.0, fov_is_vertical=True, **kwargs):
    r1 = solve_one_image_fixed_fov(catalog_path, image1_path, fov_deg, fov_is_vertical, **kwargs)
    r2 = solve_one_image_fixed_fov(catalog_path, image2_path, fov_deg, fov_is_vertical, **kwargs)
    if r1 is None or r2 is None:
        print("[fixedFOV] Could not get both attitudes."); 
        return r1, r2, None
    sep = angle_between_vecs_deg(r1["boresight"], r2["boresight"])
    print(f"[fixedFOV] Boresight separation (center-to-center) = {sep:.4f}°")
    return r1, r2, sep


# ============================ SINGLE-IMAGE CONVENIENCE CALL ============================

def solve_single_image_fixed_fov(catalog_path,
                                 image_path,
                                 *,
                                 fov_deg=66.0,           # set your known vertical (or horizontal) FoV
                                 fov_is_vertical=True,   # True if fov_deg is vertical FoV; False if it's horizontal
                                 mag_limit=3,            # bright seed catalog limit (same as before)
                                 max_image_stars=26,
                                 k_desc=5,
                                 top_m=10,
                                 ang_gate_deg=0.45,
                                 px_gate_seed=22.0,
                                 px_gate_assign=20.0,
                                 refine_loops=1,
                                 seed=7,
                                 annotate_suffix="_fixedFOV.png"):
    """
    Runs the fixed-FOV, center-anchored attitude solve on ONE image.
    Prints RA/Dec and returns a dict with pose, matches, etc.
    """
    res = solve_one_image_fixed_fov(
        catalog_path, image_path, fov_deg,
        fov_is_vertical=fov_is_vertical,
        mag_limit=mag_limit, max_image_stars=max_image_stars,
        k_desc=k_desc, top_m=top_m,
        ang_gate_deg=ang_gate_deg,
        px_gate_seed=px_gate_seed,
        px_gate_assign=px_gate_assign,
        refine_loops=refine_loops,
        seed=seed,
        annotate_suffix=annotate_suffix
    )

    if res is None:
        print("[single-fixedFOV] Failed to solve attitude for this image.")
        return None

    print("[single-fixedFOV] Done.")
    print(f"  RA  = {res['ra_deg']:.4f}°")
    print(f"  Dec = {res['dec_deg']:.4f}°")
    print(f"  Matched {len(res['matches'])} stars (of {len(res['stars'])} used)")

    return res

# ---------------- Example usage (uncomment to run directly) ----------------
if __name__ == "__main__":
    catalog_txt = r"C:\Users\kiira\OneDrive\Desktop\Space Challenges\N_Approach\hipparcos_N3.txt"
    image_png   = r"C:\Users\kiira\OneDrive\Desktop\Space Challenges\N_Approach\stellarium_image.png"
    solve_single_image_fixed_fov(
        catalog_txt, image_png,
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
        annotate_suffix="_fixedFOV.png"
    )



# # ---------------- Example call (center-anchored) ----------------
# if __name__ == "__main__":
#     catalog_txt = r"C:\Users\kiira\OneDrive\Desktop\Space Challenges\N_Approach\hipparcos_N3.txt"
#     image1_png  = r"C:\Users\kiira\OneDrive\Desktop\Space Challenges\N_Approach\pov1.png"
#     image2_png  = r"C:\Users\kiira\OneDrive\Desktop\Space Challenges\N_Approach\pov2.png"

#     # For your Stellarium setup (stereographic), pass the known vertical FOV if that's what you set.
#     solve_two_images_fixed_fov(
#         catalog_txt, image1_png, image2_png,
#         fov_deg=66.0,          # put your actual vertical FoV here
#         fov_is_vertical=True,  # set False if your 66° is horizontal instead
#         mag_limit=3, max_image_stars=26,
#         k_desc=5, top_m=10,
#         ang_gate_deg=0.45,
#         px_gate_seed=22.0,
#         px_gate_assign=20.0,
#         refine_loops=1,
#         seed=7
#     )
