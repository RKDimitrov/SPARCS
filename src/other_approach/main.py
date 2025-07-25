import os
import matplotlib.pyplot as plt
from detection.detection import detect_star_centroids
from detection.vector_conversion import pixel_to_unit_vectors, compute_focal_length_px
from catalog.catalog import load_hipparcos_catalog
from matching.matching import match_stars
from attitude.wahba import solve_wahba
from utils.io import print_matched_stars, save_matched_stars_to_csv

IMAGE_PATH = "../images/MatchingImage.png"
CATALOG_PATH = "../HipparcosCatalog.txt"
OUTPUT_MATCHED_CSV = "../outputs/matched_stars.csv"
FOV_DEG = 69.9


def main():
    # 1. Detect stars
    centroids, raw_image, binary = detect_star_centroids(IMAGE_PATH)
    print(f"Detected {len(centroids)} stars.")

    # 2. Show detection results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image with Detected Stars")
    plt.imshow(raw_image, cmap='gray')
    for (x, y) in centroids:
        plt.plot(x, y, 'ro', markersize=4)
    plt.subplot(1, 2, 2)
    plt.title("Binary Thresholded Image")
    plt.imshow(binary, cmap='gray')
    plt.show()

    # 3. Convert to unit vectors
    image_height, image_width = raw_image.shape
    focal_length_px = compute_focal_length_px(image_width, FOV_DEG)
    unit_vectors_cam = pixel_to_unit_vectors(centroids, raw_image.shape, focal_length_px)
    print(f"Estimated focal length: {focal_length_px:.2f} px")
    print(f"First 5 unit vectors in camera frame:\n{unit_vectors_cam[:5]}")

    # 4. Load catalog
    catalog_df = load_hipparcos_catalog(CATALOG_PATH)

    # 5. Match stars
    catalog_vecs, matched_indices = match_stars(unit_vectors_cam, catalog_df, max_angle_deg=5.0)
    if len(matched_indices) >= 3:
        print_matched_stars(unit_vectors_cam, catalog_df, matched_indices)
        # 6. Wahba solver
        cam_vecs = unit_vectors_cam[:len(matched_indices)]
        inertial_vecs = catalog_vecs
        R_cam_inertial = solve_wahba(cam_vecs, inertial_vecs)
        print("\nRotation Matrix (Camera to Inertial):\n", R_cam_inertial)
    else:
        print("Not enough matches to determine orientation.")

    # 7. Save results
    save_matched_stars_to_csv(OUTPUT_MATCHED_CSV, unit_vectors_cam, catalog_df, matched_indices)

if __name__ == "__main__":
    main() 