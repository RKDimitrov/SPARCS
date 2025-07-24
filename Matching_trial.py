# matching.py
# This script performs efficient matching of detected star triad properties to catalog triad properties
# using a KDTree for nearest-neighbor search. It assumes 'catalog_triad_properties.csv' and 
# 'detected_star_properties.csv' exist from previous runs of catalog and detection scripts.

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from scipy.stats import mode  # For consensus voting

# Load catalog triad properties
catalog_df = pd.read_csv(r'catalog_triad_properties_n.csv')
catalog_props = catalog_df[['perimeter', 'area', 'polar_moment', 'side_ratio', 'normal_coefficient']].values

# Normalize catalog properties
scaler = MinMaxScaler()
catalog_props_norm = scaler.fit_transform(catalog_props)

# Build KDTree on normalized catalog properties
catalog_tree = KDTree(catalog_props_norm)

# Load detected triad properties
detected_df = pd.read_csv(r'.\detected_star_properties_n2.csv')
detected_props = detected_df[['perimeter', 'area', 'polar_moment', 'side_ratio', 'normal_coefficient']].values

# Normalize detected using same scaler
detected_props_norm = scaler.transform(detected_props)

# Match: For each detected triad, find nearest catalog triad(s)
matches = []
tol = 0.05  # Distance threshold; tune based on noise/tests
k_neighbors = 5  # Check top 5 nearest for better robustness

for i, det_prop in enumerate(detected_props_norm):
    dists, idxs = catalog_tree.query(det_prop, k=k_neighbors)
    candidates = []
    for dist, idx in zip(dists, idxs):
        if dist < tol:
            cat_triad = eval(catalog_df.iloc[idx]['triad'])  # Convert string tuple to actual tuple
            det_triad = eval(detected_df.iloc[i]['triad'])
            candidates.append({
                'detected_triad': det_triad,
                'catalog_triad': cat_triad,
                'distance': dist
            })
    if candidates:
        # Select best (lowest dist) or vote; here, add all for voting later
        matches.extend(candidates)

# Convert matches to DataFrame
matches_df = pd.DataFrame(matches)
print("Matched Triads:")
print(matches_df)

# Save matches
matches_df.to_csv(r'.\matched_triads_n2.csv', index=False)
print("Matches saved to .\matched_triads.csv")

# Identify Stars: Voting across matches for consistent mapping
star_mapping = defaultdict(list)  # detected_idx -> list of catalog_ids
for match in matches:
    det_ids = match['detected_triad']
    cat_ids = match['catalog_triad']
    for d, c in zip(det_ids, cat_ids):
        star_mapping[d].append(c)

# In matching.py, add this after import statements
hip_catalog = pd.read_csv(r'.\HipparcosCatalog.txt', sep='|')
hip_catalog.columns = hip_catalog.columns.str.strip()
hip_catalog['hip_id'] = hip_catalog['name'].str.split(' ').str[1]  # Extract HIP number from 'name'

# Replace the identification saving section with this
identified_stars = {}
for det_idx, cat_list in star_mapping.items():
    if cat_list:
        most_common = mode(cat_list).mode
        identified_stars[det_idx] = {
            'catalog_index': most_common,
            'hip_id': hip_catalog.iloc[most_common]['hip_id']
        }

print("Identified Stars (detected_idx: {'catalog_index': id, 'hip_id': hip}):")
print(identified_stars)

# Save identified stars to CSV with HIP ID
identified_list = [{'detected_idx': k, 'catalog_index': v['catalog_index'], 'hip_id': v['hip_id']} for k, v in identified_stars.items()]
identified_df = pd.DataFrame(identified_list)
identified_df.to_csv(r'.\identified_stars_n2.csv', index=False)
print(r"Identified stars saved to .\identified_stars.csv")