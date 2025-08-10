# pair_approach/matching/pair_matching.py

import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

# ----------------- UTILITIES -----------------

def _unit_rows(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))

def _extract_hip_int(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if s.upper().startswith("HIP"):
        parts = s.replace(",", " ").split()
        for p in parts:
            if p.isdigit():
                return int(p)
    try:
        return int(s)
    except Exception:
        return s

def _debug_dump_matches(tag, matches, limit=20):
    print(f"[DEBUG] {tag}: {len(matches)} matches")
    for i, m in enumerate(matches[:limit]):
        print(f"  {i:02d}) img={m['image_star_index']}  cat_idx={m['catalog_star_index']}  hip={m['catalog_star_id']}  conf={m.get('confidence', 0):.3f}")
    if len(matches) > limit:
        print(f"  ... ({len(matches)-limit} more)")

# ----------------- 🔧 FIXED: ADD MISSING FUNCTION DEFINITION -----------------

def pair_angle_matching(catalog_vectors, image_vectors, catalog_ids, 
                       intensities=None, catalog_vmag=None,
                       max_fov_deg=25.0, tolerance_deg=0.1, 
                       top_k=10, min_votes=1,
                       use_brightness_rank=True, debug=False):
    """
    🔧 FIXED: Added missing function definition!
    
    Pair-angle matching:
      - Indicizza gli ANGOLI di coppie del catalogo entro max_fov_deg (subset opzionale).
      - Confronta angoli di coppie dell'immagine entro max_fov_deg.
      - Accumula voti su (image_star_index -> HIP), poi risolve conflitti deterministicamente.
    Ritorna una lista di dict con chiavi:
      image_star_index, catalog_star_index (indice nel catalogo COMPLETO),
      catalog_star_id (HIP), confidence
    """
    # Input as array
    catalog_vectors = np.asarray(catalog_vectors, dtype=float)
    image_vectors   = np.asarray(image_vectors, dtype=float)
    cat_ids_full    = np.asarray(catalog_ids)

    if debug:
        print("[DEBUG] pair_angle_matching: shapes",
              f"catalog_vectors={catalog_vectors.shape}, image_vectors={image_vectors.shape}, ids={len(cat_ids_full)}")
        print("[DEBUG] First 5 catalog_ids:", list(cat_ids_full[:5]))

    img_vecs = _unit_rows(image_vectors)
    cat_vecs_full = _unit_rows(catalog_vectors)
    Nimg, Ncat = len(img_vecs), len(cat_vecs_full)

    # 1) Sottoinsieme immagine (ordinamento deterministico per intensità)
    if use_brightness_rank and intensities is not None:
        intensities = np.asarray(intensities, dtype=float)
        order_img = np.lexsort((np.arange(Nimg), -intensities))
        keep_img = order_img[:min(max(top_k * 2, 40), Nimg)]
    else:
        keep_img = np.arange(Nimg)
        intensities = np.ones(Nimg, dtype=float)

    img_vecs_use = img_vecs[keep_img]
    intens_use   = intensities[keep_img]

    # 2) Sottoinsieme catalogo (ordinamento deterministico per vmag, poi HIP)
    if use_brightness_rank and catalog_vmag is not None:
        catalog_vmag = np.asarray(catalog_vmag, dtype=float)
        hips_int_full = np.array([_extract_hip_int(h) for h in cat_ids_full])
        order_cat = np.lexsort((hips_int_full, catalog_vmag))
        sel_cat = order_cat[:min(6000, Ncat)]
    else:
        sel_cat = np.arange(Ncat)

    cat_vecs = cat_vecs_full[sel_cat]
    cat_ids  = cat_ids_full[sel_cat]
    hips_int_local = np.array([_extract_hip_int(h) for h in cat_ids])

    # mapping subset->indice completo
    cat_local_to_global = {int(i_local): int(sel_cat[i_local]) for i_local in range(len(sel_cat))}

    if debug:
        print(f"[DEBUG] Using subset: img={len(img_vecs_use)}/{Nimg}  cat={len(cat_vecs)}/{Ncat}")
        print("[DEBUG] First 5 subset cat_ids:", list(cat_ids[:5]))

    # 3) Indicizza coppie catalogo entro FOV (metrica chordale)
    max_chord = 2.0 * np.sin(np.radians(max_fov_deg) / 2.0)
    ct = KDTree(cat_vecs)

    cat_pairs_angles = []
    cat_pairs_ij = []
    for i in range(len(cat_vecs)):
        nbrs = ct.query_ball_point(cat_vecs[i], r=max_chord)
        nbrs = sorted([j for j in nbrs if j > i])
        for j in nbrs:
            ang = _angle_deg(cat_vecs[i], cat_vecs[j])
            cat_pairs_angles.append(ang)
            cat_pairs_ij.append((i, j))
    if len(cat_pairs_angles) == 0:
        if debug:
            print("[DEBUG] Nessuna coppia nel catalogo entro FOV.")
        return []

    cat_pairs_angles = np.asarray(cat_pairs_angles, dtype=float)
    kdt = KDTree(cat_pairs_angles.reshape(-1, 1))

    if debug:
        print(f"[DEBUG] Catalog pairs indexed: {len(cat_pairs_angles)}")

    # 4) Genera coppie immagine e accumula voti
    it = KDTree(img_vecs_use)
    votes = defaultdict(float)
    pairs_considered = 0

    for a in range(len(img_vecs_use)):
        nbrs = it.query_ball_point(img_vecs_use[a], r=max_chord)
        nbrs = sorted([b for b in nbrs if b > a])
        for b in nbrs:
            pairs_considered += 1
            ang_ab = _angle_deg(img_vecs_use[a], img_vecs_use[b])
            match_idxs = kdt.query_ball_point([[ang_ab]], r=tolerance_deg)[0]
            if not match_idxs:
                continue
            w = float(min(intens_use[a], intens_use[b]))
            ia = int(keep_img[a]); ib = int(keep_img[b])
            for k in match_idxs:
                i, j = cat_pairs_ij[k]
                hip_i = int(hips_int_local[i]) if isinstance(hips_int_local[i], (int, np.integer)) else hips_int_local[i]
                hip_j = int(hips_int_local[j]) if isinstance(hips_int_local[j], (int, np.integer)) else hips_int_local[j]
                votes[(ia, hip_i)] += w
                votes[(ib, hip_j)] += w
                votes[(ia, hip_j)] += w
                votes[(ib, hip_i)] += w

    if debug:
        print(f"[DEBUG] Image pairs considered: {pairs_considered}")
        print(f"[DEBUG] Vote entries: {len(votes)}")

    if not votes:
        if debug:
            print("[DEBUG] Nessun voto: aumenta top_k o tolerance_deg.")
        return []

    def hip_sortkey(h):  # per il tie-break deterministico
        return h if isinstance(h, (int, np.integer)) else 10**9

    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0][0], hip_sortkey(kv[0][1])))

    used_img, used_hip = set(), set()
    final = []
    for (img_idx, hip), score in ranked:
        if img_idx in used_img or hip in used_hip:
            continue
        # trova indice locale nel subset cat
        loc_candidates = np.where(hips_int_local == hip)[0] if isinstance(hip, (int, np.integer)) \
                         else np.where(cat_ids == hip)[0]
        if loc_candidates.size == 0:
            if debug:
                print(f"[DEBUG] HIP {hip} non nel subset locale, salto")
            continue
        i_local  = int(loc_candidates[0])
        i_global = cat_local_to_global[i_local]
        final.append({
            "image_star_index": int(img_idx),
            "catalog_star_index": int(i_global),     # indice nel catalogo COMPLETO
            "catalog_star_id": cat_ids_full[i_global],
            "confidence": float(score)
        })
        used_img.add(img_idx)
        used_hip.add(hip)
        if len(final) >= top_k:
            break

    final = [m for m in final if m["confidence"] >= float(min_votes)]
    final.sort(key=lambda m: (m["image_star_index"], _extract_hip_int(m["catalog_star_id"])))

    if debug:
        _debug_dump_matches("PAIR final", final)

    return final

# ----------------- TRIAD "LEGGERO" -----------------

def triad_refinement(image_vectors, catalog_vectors, initial_matches,
                     side_tol_deg=1.0, vote_threshold=1, angle_tol_deg=None, debug=False):
    if not initial_matches:
        return []
    refined = []
    for m in initial_matches:
        triad_votes = max(1, int(round(m.get("confidence", 1.0) * 5.0)))
        mm = dict(m)
        mm["triad_votes"] = triad_votes
        refined.append(mm)
    refined = [m for m in refined if m["triad_votes"] >= int(vote_threshold)]
    refined.sort(key=lambda m: (-m["triad_votes"], m["image_star_index"], _extract_hip_int(m["catalog_star_id"])))
    if debug:
        print(f"[DEBUG] TRIAD kept {len(refined)}/{len(initial_matches)}")
        _debug_dump_matches("TRIAD final", refined)
    return refined