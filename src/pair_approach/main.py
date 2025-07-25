import os
from pair_approach.detection.image_loading import load_image
from pair_approach.detection.star_detection import detect_stars
from pair_approach.detection.vector_conversion import centroids_to_vectors
from pair_approach.detection.visualization import save_grayscale, plot_detections
from pair_approach.catalog.catalog_loader import load_hipparcos_catalog, add_catalog_unit_vectors
from pair_approach.catalog.catalog_vectorizer import get_catalog_vectors_and_ids
from pair_approach.matching.pair_matching import pair_angle_matching_with_ids
from pair_approach.utils.io import save_vectors_to_csv
from attitude_determination.compute import calculate_attitude, print_results

# --- CONFIG ---
IMAGE_PATH = os.path.join(os.path.dirname(__file__), '../images/MatchingImage.png')
CATALOG_PATH = os.path.join(os.path.dirname(__file__), '../HipparcosCatalog.txt')
OUTPUT_VECTOR_CSV = os.path.join(os.path.dirname(__file__), '../outputs/star_vectors.csv')
FOV_DEG = 30
QUEST_MEASUREMENTS_FILE = os.path.join(os.path.dirname(__file__), '../outputs/quest_measurements.txt')

# --- PIPELINE ---
def main():
    img, img_pil = load_image(IMAGE_PATH)
    centroids, intensities = detect_stars(img)
    print(f"Detected {len(centroids)} stars")
    if len(centroids) > 0:
        print("Centroids (y, x):", centroids)
        print("Intensities:", intensities)
    save_grayscale(img_pil, os.path.join(os.path.dirname(__file__), '../outputs/grayscale.png'))
    plot_detections(img, centroids, os.path.join(os.path.dirname(__file__), '../outputs/detected_stars.png'))

    # 2. Convert centroids to vectors
    img_height, img_width = img.shape
    star_vectors = centroids_to_vectors(centroids, img_height, img_width, fov_deg=FOV_DEG)
    print("Star Vectors:\n", star_vectors)
    save_vectors_to_csv(star_vectors, OUTPUT_VECTOR_CSV)

    # 3. Load catalog and process
    hip = load_hipparcos_catalog(CATALOG_PATH)
    hip = add_catalog_unit_vectors(hip)
    catalog_vectors, catalog_ids = get_catalog_vectors_and_ids(hip, id_col='name')

    # 4. Match detected stars to catalog
    matches = pair_angle_matching_with_ids(catalog_vectors, catalog_ids, star_vectors, max_fov_deg=FOV_DEG)
    print("\nStar Matches:")
    for match in matches:
        print(f"Image star {match['image_star_index']} matched to catalog star {match['catalog_star_id']} (votes: {match['votes']})")

    # 5. Prepare QUEST input and run attitude determination
    # Write matched star vectors and catalog IDs to a temporary file for QUEST
    if matches:
        with open(QUEST_MEASUREMENTS_FILE, 'w') as f:
            f.write("x\ty\tz\tHIP_ID\n")
            for match in matches:
                idx = match['image_star_index']
                hip_id = match['catalog_star_id']
                x, y, z = star_vectors[idx]
                f.write(f"{x}\t{y}\t{z}\t{hip_id}\n")
        print(f"\nRunning QUEST attitude determination using {QUEST_MEASUREMENTS_FILE}...")
        results = calculate_attitude(QUEST_MEASUREMENTS_FILE, CATALOG_PATH)
        print_results(results)
    else:
        print("No matches found; skipping attitude determination.")

if __name__ == "__main__":
    main() 