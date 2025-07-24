##testing pair matching 
import numpy as np, pandas as pd

##USING FILTERED HIPPARCUS DATA OF ONLY 1000 STARS TO FIND AND CALCLATE THE DIFFERENT TRIADS FROM THE CATALOGUE WHICH CAN 
##BE USED FOR COMPARISON

#STEP 1 - EXTRACTING ONLY NECESSARY DATA
hip = pd.read_csv("HipparcosCatalog.txt", sep='|')
hip.columns = hip.columns.str.strip()
#only want relevant columns
catalog = hip[['ra', 'dec', 'vmag']].copy() #only need right ascension, declination and vmag values



#STEP 2 - FORMING THE 3D UNIT VECTORS FROM THE RA,DEC VALUES
def hms_to_deg(hms_str):
    #converting RA "HH MM SS.SSS" to degrees as its given in hours minutes seconds HMS
    h, m, s = [float(part) for part in hms_str.split()]
    return 15 * (h + m/60 + s/3600) #formula for conversion

def dms_to_deg(dms_str):
    #converting DEC "±DD MM SS.SSS" to degrees as dec given in degress minutes seconds
    parts = dms_str.split()
    sign = -1 if parts[0].startswith('-') else 1
    #removing sign from degrees part to convert abs 
    deg = float(parts[0].replace('+','').replace('-',''))
    m = float(parts[1])
    s = float(parts[2])
    return sign * (deg + m/60 + s/3600)

#using the functions previously created for the conversions
hip['ra_deg'] = hip['ra'].apply(hms_to_deg)
hip['dec_deg'] = hip['dec'].apply(dms_to_deg)

#converting to radians
hip['ra_rad'] = np.deg2rad(hip['ra_deg'])
hip['dec_rad'] = np.deg2rad(hip['dec_deg'])

#calculating unit vectors
hip['x'] = np.cos(hip['dec_rad']) * np.cos(hip['ra_rad'])
hip['y'] = np.cos(hip['dec_rad']) * np.sin(hip['ra_rad'])
hip['z'] = np.sin(hip['dec_rad'])

#checking coordinates, checked and close to 1, these are based on the celestial coordinate system
print(hip[['x', 'y', 'z']].head())

#creating 3D plot in celestial coordinate system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hip['x'], hip['y'], hip['z'], s=1)
ax.set_title("3D Celestial Star Positions")
#plt.show()

from scipy.spatial import KDTree
import itertools
import numpy as np
import pandas as pd




##STEP 3 - calculating the triads from the catalog but filtering by stars close together in the camera's FoV using KDTree
vectors = hip[['x', 'y', 'z']].values

def angular_distance(v1, v2):
    dot = np.dot(v1, v2)
    return np.arccos(np.clip(dot, -1, 1))

MAX_ANGLE_DEG = 69.9  # Field of View threshold in degrees of camera
MAX_ANGLE_RAD = np.deg2rad(MAX_ANGLE_DEG)

#Build KDTree for fast neighbor search in 3D space
tree = KDTree(vectors)


catalog_vectors = hip[['x', 'y', 'z']].values
image_star_df = pd.read_csv("star-vectors.csv")  # contains 'x', 'y', 'z' columns
image_vectors = image_star_df[['x', 'y', 'z']].values

import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
from collections import defaultdict

def pair_angle_matching_with_ids(catalog_vectors, catalog_ids, image_vectors, 
                                max_fov_deg=30, tolerance_deg=0.01):
    """
    Match stars from image to catalog using pairwise angular distances, returning HIP IDs.

    Args:
      catalog_vectors: np.ndarray, shape (N_cat, 3) - unit vectors for catalog stars
      catalog_ids: list or np.ndarray or pd.Series - catalog star IDs (e.g. HIP numbers)
      image_vectors: np.ndarray, shape (N_img, 3) - unit vectors for image stars
      max_fov_deg: float - max allowed angular separation between pairs (camera FoV)
      tolerance_deg: float - angular tolerance for matching pairs

    Returns:
      final_matches: list of dicts
        [{'image_star_index': i_img, 'catalog_star_index': i_cat, 'catalog_star_id': hip_id, 'votes': vote_count}, ...]
    """
    import numpy as np
    from scipy.spatial import KDTree
    from itertools import combinations
    from collections import defaultdict

    max_fov_rad = np.deg2rad(max_fov_deg)
    tolerance_rad = np.deg2rad(tolerance_deg)

    catalog_pairs = []
    catalog_pair_angles = []

    for i, j in combinations(range(len(catalog_vectors)), 2):
        dot = np.dot(catalog_vectors[i], catalog_vectors[j])
        dot = np.clip(dot, -1, 1)
        angle = np.arccos(dot)
        if angle <= max_fov_rad:
            catalog_pairs.append((i, j))
            catalog_pair_angles.append([angle])

    catalog_pair_angles = np.array(catalog_pair_angles)
    catalog_tree = KDTree(catalog_pair_angles)

    image_pairs = []
    image_pair_angles = []

    for i, j in combinations(range(len(image_vectors)), 2):
        dot = np.dot(image_vectors[i], image_vectors[j])
        dot = np.clip(dot, -1, 1)
        angle = np.arccos(dot)
        if angle <= max_fov_rad:
            image_pairs.append((i, j))
            image_pair_angles.append([angle])

    image_pair_angles = np.array(image_pair_angles)

    match_votes = defaultdict(int)

    for idx, angle_feat in enumerate(image_pair_angles):
        nearby_indices = catalog_tree.query_ball_point(angle_feat, r=tolerance_rad)
        i_img, j_img = image_pairs[idx]

        for cat_idx in nearby_indices:
            i_cat, j_cat = catalog_pairs[cat_idx]
            match_votes[(i_img, i_cat)] += 1
            match_votes[(j_img, j_cat)] += 1

    vote_table = defaultdict(lambda: defaultdict(int))
    for (img_idx, cat_idx), votes in match_votes.items():
        vote_table[img_idx][cat_idx] += votes

    final_matches = []
    for img_idx, cat_votes in vote_table.items():
        best_cat_idx = max(cat_votes, key=cat_votes.get)
        final_matches.append({
            'image_star_index': img_idx,
            'catalog_star_index': best_cat_idx,
            'catalog_star_id': catalog_ids[best_cat_idx],
            'votes': cat_votes[best_cat_idx]
        })

    return final_matches


# Example assuming you have:
# hip['hip_id'] = HIP IDs from your catalog dataframe (same order as catalog_vectors)
catalog_ids = hip['name'].values  

matches = pair_angle_matching_with_ids(catalog_vectors, catalog_ids, image_vectors)

for match in matches:
    print(f"Image star {match['image_star_index']} matched to catalog star {match['catalog_star_id']}")
