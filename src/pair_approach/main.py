import os, random, numpy as np
from pair_approach.detection.image_loading import load_image
from pair_approach.detection.star_detection import detect_stars
from pair_approach.detection.vector_conversion import centroids_to_vectors, test_coordinate_systems
from pair_approach.detection.visualization import (
    save_grayscale, plot_detections, plot_matched_stars,
    create_star_catalog_overlay
)
# Fixed imports - use the classes you actually have
from pair_approach.catalog.catalog import HipparcosCatalog  # Use your HipparcosCatalog class
from pair_approach.matching.pair_matching import pair_angle_matching, triad_refinement  # Use your function name
from pair_approach.utils.io import save_vectors_to_csv
from attitude_determination.compute import calculate_attitude, print_results
from pair_approach.matching.ransac_attitude import ransac_refine_matches

# ------------- FIXED CONFIG -------------
SEED = 42
IMAGE_PATH = os.path.join(os.path.dirname(__file__), '../images/MatchingImage4.png')
CATALOG_PATH = os.path.join(os.path.dirname(__file__), '../HipparcosCatalog.txt')
OUTPUT_VECTOR_CSV = os.path.join(os.path.dirname(__file__), '../outputs/star_vectors.csv')

# 🔧 FIXED: Reduced FOV and proper parameters
FOV_DEG = 25  # REDUCED from 66! Start smaller for testing
FOV_DIRECTION = 'vertical'      # You said vertical FOV
COORDINATE_SYSTEM = 'image'     # Y down, X right (standard image coordinates)

QUEST_MEASUREMENTS_FILE = os.path.join(os.path.dirname(__file__), '../outputs/quest_measurements.txt')
POLARIS_HIP = 11767  # per debug

def get_catalog_vectors_and_ids(catalog):
    """
    Extract unit vectors and HIP IDs from catalog
    Fixed function that was missing from your imports
    """
    vectors = []
    hip_ids = []
    
    for _, row in catalog.catalog.iterrows():  # Use catalog.catalog to access the DataFrame
        ra_deg = row['RA_deg']
        dec_deg = row['Dec_deg']
        hip_id = row['HIP']
        
        # Convert to unit vector
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        
        vectors.append([x, y, z])
        hip_ids.append(hip_id)
    
    return np.array(vectors), np.array(hip_ids)

def report_star(label_hip, matches, tag=""):
    hips = [int(m['catalog_star_id']) if str(m['catalog_star_id']).isdigit() else m['catalog_star_id'] for m in matches]
    idxs = [i for i, h in enumerate(hips) if h == label_hip]
    print(f"[DEBUG] {tag} HIP {label_hip} present={len(idxs)>0} indices={idxs}")

def dump_matches(tag, matches, limit=25):
    print(f"[DEBUG] {tag}: {len(matches)} matches")
    for i, m in enumerate(matches[:limit]):
        print(f"  {i:02d}) img={m['image_star_index']}  cat_idx={m['catalog_star_index']}  hip={m['catalog_star_id']}  conf={m.get('confidence',0):.3f}")
    if len(matches) > limit:
        print(f"  ... ({len(matches)-limit} more)")

def main():
    # 0) Determinismo totale
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)

    print("=== STAR TRACKER - CORRECTED VERSION ===\n")
    print(f"🔧 FIXED CONFIG: FOV={FOV_DEG}°, direction={FOV_DIRECTION}, coords={COORDINATE_SYSTEM}")

    # 1) DETECTION
    print("1) Loading image and detecting stars ...")
    img, img_pil = load_image(IMAGE_PATH)
    centroids, intensities = detect_stars(img)
    print(f"   Detected {len(centroids)} stars")
    if len(centroids) == 0:
        print("   ERROR: No stars detected.")
        return

    # Debug: Print some centroid coordinates
    print(f"   First 3 centroids (y,x): {centroids[:3].tolist()}")
    print(f"   Image shape (H,W): {img.shape}")

    # Ordina i centri per stabilità (y,x)
    order = sorted(range(len(centroids)), key=lambda i: (centroids[i][1], centroids[i][0]))
    centroids = np.array([centroids[i] for i in order], dtype=float)
    intensities = np.array([intensities[i] for i in order], dtype=float)

    output_dir = os.path.join(os.path.dirname(__file__), '../outputs')
    os.makedirs(output_dir, exist_ok=True)

    save_grayscale(img_pil, os.path.join(output_dir, 'grayscale.png'))
    plot_detections(img, centroids, os.path.join(output_dir, 'detected_stars.png'))

    # 2) VECTOR CONVERSION - 🔧 FIXED WITH PROPER PARAMETERS
    print("2) Converting centroids to unit vectors ...")
    img_height, img_width = img.shape
    
    # 🔧 TEST DIFFERENT COORDINATE SYSTEMS FIRST
    print("   Testing coordinate systems...")
    test_coordinate_systems(centroids, img_height, img_width, FOV_DEG)
    
    # 🔧 USE PROPER PARAMETERS
    star_vectors = centroids_to_vectors(
        centroids, img_height, img_width, FOV_DEG,
        fov_direction=FOV_DIRECTION,      # CRITICAL: Specify vertical FOV
        coordinate_system=COORDINATE_SYSTEM  # CRITICAL: Specify image coordinates
    )
    print(f"   Produced {len(star_vectors)} unit vectors")
    print(f"   Vector magnitudes: {[np.linalg.norm(v) for v in star_vectors[:3]]}")  # Should be ~1.0
    save_vectors_to_csv(star_vectors, OUTPUT_VECTOR_CSV)

    # 3) CATALOG LOADING - 🔧 FIXED TO USE YOUR ACTUAL CLASS
    print("3) Loading catalog ...")
    catalog = HipparcosCatalog(CATALOG_PATH)  # Use your actual class
    
    # 🔧 FIXED: Use the corrected function
    catalog_vectors, catalog_ids = get_catalog_vectors_and_ids(catalog)
    catalog_ids = np.asarray(catalog_ids, dtype=int)

    print(f"   Catalog loaded: {len(catalog_vectors)} stars")
    print(f"   Catalog vectors shape: {catalog_vectors.shape}")
    print(f"   First 5 HIP IDs: {catalog_ids[:5]}")
    
    # Check if we have reasonable catalog data
    if len(catalog_vectors) == 0:
        print("   ERROR: No catalog data loaded!")
        return
    
    # Verify unit vectors
    catalog_mags = [np.linalg.norm(v) for v in catalog_vectors[:5]]
    print(f"   First 5 catalog vector magnitudes: {catalog_mags}")  # Should be ~1.0

    # Polaris check
    polaris_present = int(POLARIS_HIP) in set(catalog_ids.tolist())
    print(f"   Polaris HIP {POLARIS_HIP} in catalog: {polaris_present}")

    # 4) MATCHING - 🔧 FIXED FUNCTION CALL
    print("4) Pair-angle matching ...")
    
    # 🔧 FIXED: Use your actual function with proper parameters
    matches_initial = pair_angle_matching(
        catalog_vectors=catalog_vectors,
        image_vectors=star_vectors,
        catalog_ids=catalog_ids,
        intensities=intensities,
        catalog_vmag=None,  # You don't have vmag in your catalog class
        max_fov_deg=FOV_DEG,
        tolerance_deg=0.2,
        use_brightness_rank=True,
        top_k=15,  # Reduced for testing
        min_votes=1.0,
        debug=True
    )
    
    dump_matches("PAIR initial", matches_initial, limit=25)
    report_star(POLARIS_HIP, matches_initial, tag="PAIR initial")

    if not matches_initial:
        print("   ❌ NO INITIAL MATCHES FOUND!")
        print("   🔧 TROUBLESHOOTING:")
        print(f"      - Try larger tolerance_deg (current: 0.2)")
        print(f"      - Try larger FOV_DEG (current: {FOV_DEG})")
        print(f"      - Check coordinate system")
        print(f"      - Verify star detection quality")
        return

    # 5) TRIAD (leggero)
    print("5) TRIAD refinement ...")
    matches_refined = triad_refinement(
        image_vectors=star_vectors,
        catalog_vectors=catalog_vectors,
        initial_matches=matches_initial,
        debug=True
    )
    dump_matches("TRIAD refined", matches_refined, limit=25)
    report_star(POLARIS_HIP, matches_refined, tag="TRIAD refined")

    # 6) RANSAC
    if len(matches_refined) >= 3:
        print("6) RANSAC refinement ...")
        matches_final = ransac_refine_matches(
            star_vectors,
            catalog_vectors,
            catalog_ids,
            seed_matches=matches_refined,
            angle_thresh_deg=1.5,
            n_iterations=200,
        )
        dump_matches("RANSAC final", matches_final, limit=25)
        report_star(POLARIS_HIP, matches_final, tag="RANSAC final")
    else:
        print("6) RANSAC skipped (too few matches)")
        matches_final = matches_refined

    # 7) VISUALIZATION
    best_matches = matches_final if matches_final else matches_refined
    if best_matches:
        print("7) Creating visualizations ...")
        plot_matched_stars(
            img, centroids, best_matches,
            save_path=os.path.join(output_dir, 'corrected_matched_stars.png')
        )
        create_star_catalog_overlay(
            img, centroids, best_matches, intensities,
            save_path=os.path.join(output_dir, 'corrected_star_overlay.png')
        )

    # 8) QUEST
    if best_matches:
        print("8) Running QUEST ...")
        with open(QUEST_MEASUREMENTS_FILE, 'w') as f:
            f.write("x\ty\tz\tHIP_ID\n")
            for match in best_matches:
                idx = match['image_star_index']
                hip_id = int(match['catalog_star_id'])
                x, y, z = star_vectors[idx]
                f.write(f"{x:.6f}\t{y:.6f}\t{z:.6f}\t{hip_id}\n")

        try:
            quest_results = calculate_attitude(QUEST_MEASUREMENTS_FILE, CATALOG_PATH)
            if quest_results:
                print_results(quest_results)
                mean_err = quest_results.get('mean_error', quest_results.get('Mean error', 999.0))
                print(f"   ✅ QUEST Mean Error: {mean_err:.3f}°")
                
                if mean_err < 2.0:
                    print("   🏆 EXCELLENT SOLUTION!")
                elif mean_err < 10.0:
                    print("   ✅ GOOD SOLUTION!")
                else:
                    print("   ⚠️  NEEDS IMPROVEMENT")
            else:
                print("   ❌ QUEST failed to produce results")
        except Exception as e:
            print(f"   ❌ QUEST failed: {e}")

    # 9) SUMMARY
    print("\n" + "="*60)
    print("CORRECTED STAR TRACKER SUMMARY")
    print("="*60)
    print(f"Configuration: FOV={FOV_DEG}° ({FOV_DIRECTION}), coords={COORDINATE_SYSTEM}")
    print(f"Stars detected: {len(centroids)}")
    print(f"Initial matches: {len(matches_initial) if 'matches_initial' in locals() else 0}")
    print(f"Refined matches: {len(matches_refined) if 'matches_refined' in locals() else 0}")
    print(f"Final matches: {len(matches_final) if 'matches_final' in locals() else 0}")
    
    if best_matches:
        hips = [int(m['catalog_star_id']) for m in best_matches]
        print(f"Unique HIP IDs: {len(set(hips))}/{len(hips)}")
        print("✅ Check outputs folder for visualizations")
    else:
        print("❌ No matches found")
        print("🔧 Try:")
        print("   - Increase FOV_DEG")
        print("   - Increase tolerance_deg")
        print("   - Test different coordinate_system")
        print("   - Check star detection parameters")

if __name__ == "__main__":
    main()