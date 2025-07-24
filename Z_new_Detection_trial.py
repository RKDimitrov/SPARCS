import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_star_centroids(image_path, threshold=200):
    # Load and process image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Thresholding to get binary stars
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Find contours/blobs
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centroids.append((cx, cy))

    return centroids, image, binary

# Example usage
image_path = "MatchingImage.png"  # replace with your actual image path
centroids, raw_image, binary = detect_star_centroids(image_path)

# Show result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Z_Original Image with Detected Stars")
plt.imshow(raw_image, cmap='gray')
for (x, y) in centroids:
    plt.plot(x, y, 'ro', markersize=4)

plt.subplot(1, 2, 2)
plt.title("Z_Binary Thresholded Image")
plt.imshow(binary, cmap='gray')
plt.show()

print(f"Detected {len(centroids)} stars.")


def pixel_to_unit_vectors(centroids, image_shape, focal_length_px):
    """
    Convert pixel coordinates to unit vectors in the camera frame.
    """
    h, w = image_shape
    cx, cy = w / 2, h / 2  # Principal point

    unit_vectors = []
    for (u, v) in centroids:
        x = u - cx
        y = v - cy
        z = focal_length_px
        vec = np.array([x, y, z])
        unit_vec = vec / np.linalg.norm(vec)
        unit_vectors.append(unit_vec)
    
    return np.array(unit_vectors)

def compute_focal_length_px(image_width, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    return (image_width / 2) / np.tan(fov_rad / 2)


# Example usage
image_height, image_width = raw_image.shape
focal_length_px = compute_focal_length_px(image_width, 69.9)

unit_vectors_cam = pixel_to_unit_vectors(centroids, raw_image.shape, focal_length_px)
print(f"Estimated focal length: {focal_length_px:.2f} px")

#unit_vectors_cam = pixel_to_unit_vectors(centroids, raw_image.shape)
print(f"First 5 unit vectors in camera frame:\n{unit_vectors_cam[:5]}")


import pandas as pd

def hms_to_deg(hms_str):
    h, m, s = map(float, hms_str.strip().split())
    return 15 * (h + m / 60 + s / 3600)

def dms_to_deg(dms_str):
    sign = -1 if dms_str.strip().startswith('-') else 1
    parts = dms_str.strip().replace('+', '').replace('-', '').split()
    d, m, s = map(float, parts)
    return sign * (d + m / 60 + s / 3600)

def load_hipparcos_catalog(txt_path, mag_limit=7):
    """
    Load Hipparcos TXT catalog, convert RA/Dec from H:M:S and D:M:S to decimal degrees.
    Filters by visual magnitude ('vmag').
    """
    df = pd.read_csv(txt_path, sep='|', engine='python')

    # Strip whitespace from column names and convert values
    df.columns = [c.strip() for c in df.columns]
    df['ra_deg'] = df['ra'].apply(hms_to_deg)
    df['dec_deg'] = df['dec'].apply(dms_to_deg)

    df['vmag'] = pd.to_numeric(df['vmag'], errors='coerce')
    df = df[df['vmag'] <= mag_limit].copy()

    # Convert to unit vectors
    ra_rad = np.deg2rad(df['ra_deg'])
    dec_rad = np.deg2rad(df['dec_deg'])

    df['x'] = np.cos(dec_rad) * np.cos(ra_rad)
    df['y'] = np.cos(dec_rad) * np.sin(ra_rad)
    df['z'] = np.sin(dec_rad)

    return df


def match_stars(image_vecs, catalog_df, max_angle_deg=5.0):
    """
    Brute-force angular matching of each image vector to closest catalog star.
    Returns matched catalog vectors and their indices.
    """
    max_angle_rad = np.deg2rad(max_angle_deg)
    matched_catalog_vecs = []
    matched_indices = []

    catalog_vecs = catalog_df[['x', 'y', 'z']].values

    for vec in image_vecs:
        dots = catalog_vecs @ vec  # cosine of angle
        angles = np.arccos(np.clip(dots, -1, 1))
        min_idx = np.argmin(angles)

        if angles[min_idx] < max_angle_rad:
            matched_catalog_vecs.append(catalog_vecs[min_idx])
            matched_indices.append(min_idx)

    return np.array(matched_catalog_vecs), matched_indices


def solve_wahba(cam_vecs, inertial_vecs):
    """
    Solves Wahba's problem using SVD.
    Returns rotation matrix from inertial to camera frame.
    """
    B = cam_vecs.T @ inertial_vecs
    U, _, Vt = np.linalg.svd(B)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def print_matched_stars(unit_vectors_cam, catalog_df, matched_indices):
    print("\nMatched Detected Stars:\n")
    print(f"{'Img#':<5} {'Unit Vector (Camera Frame)':<45} {'HIP ID':<10} {'Catalog Unit Vector':<45}")
    print("-" * 110)

    for i, idx in enumerate(matched_indices):
        cam_vec = unit_vectors_cam[i]
        hip_id = catalog_df.iloc[idx]['name'].strip()
        cat_vec = catalog_df.iloc[idx][['x', 'y', 'z']].values

        print(f"{i:<5} [{cam_vec[0]: .4f}, {cam_vec[1]: .4f}, {cam_vec[2]: .4f}]   "
              f"{hip_id:<10} [{cat_vec[0]: .4f}, {cat_vec[1]: .4f}, {cat_vec[2]: .4f}]")

import csv

def save_matched_stars_to_csv(filename, unit_vectors_cam, catalog_df, matched_indices):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x_cam", "y_cam", "z_cam", "HIP_ID"])

        for i, idx in enumerate(matched_indices):
            cam_vec = unit_vectors_cam[i]
            hip_id = catalog_df.iloc[idx]['name'].strip()
            writer.writerow([cam_vec[0], cam_vec[1], cam_vec[2],hip_id])

    print(f"Saved matched stars to '{filename}'")

# Load Hipparcos catalog
catalog_path = "HipparcosCatalog.txt"  # update with your actual file
catalog_df = load_hipparcos_catalog(catalog_path)

# Match image vectors with catalog stars
catalog_vecs, matched_idxs = match_stars(unit_vectors_cam, catalog_df)
catalog_vecs, matched_indices = match_stars(unit_vectors_cam, catalog_df, max_angle_deg=5.0)

if len(matched_indices) >= 3:
    print_matched_stars(unit_vectors_cam, catalog_df, matched_indices)

    # Proceed to Wahba solver...
    cam_vecs = np.array(unit_vectors_cam[:len(matched_indices)])
    inertial_vecs = np.array(catalog_vecs)
    R_cam_inertial = solve_wahba(cam_vecs, inertial_vecs)

    print("\nRotation Matrix (Camera to Inertial):\n", R_cam_inertial)
else:
    print("Not enough matches to determine orientation.")

save_matched_stars_to_csv("matched_stars.csv", unit_vectors_cam, catalog_df, matched_indices)
