import csv
import pandas as pd

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

def save_matched_stars_to_csv(filename, unit_vectors_cam, catalog_df, matched_indices):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x_cam", "y_cam", "z_cam", "HIP_ID"])
        for i, idx in enumerate(matched_indices):
            cam_vec = unit_vectors_cam[i]
            hip_id = catalog_df.iloc[idx]['name'].strip()
            writer.writerow([cam_vec[0], cam_vec[1], cam_vec[2], hip_id])
    print(f"Saved matched stars to '{filename}'") 