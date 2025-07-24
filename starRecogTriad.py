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

MAX_ANGLE_DEG = 30  # Field of View threshold in degrees of camera
MAX_ANGLE_RAD = np.deg2rad(MAX_ANGLE_DEG)

#Build KDTree for fast neighbor search in 3D space
tree = KDTree(vectors)



##STEP 4 - using the triads to find 5 main triangular properties: perimeter, area, polar moment, ratio of shortest to longest side
##length and the normal coefficient

def project_to_plane(points):  # since properties are planar but points are 3D, project onto plane
    centroid = np.mean(points, axis=0)  # centroid of 3 points = origin of 2D plane
    normal = centroid / np.linalg.norm(centroid)  # normal vector to projection plane (tangent to unit sphere)

    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])  # vector not parallel to normal
    
    axis1 = np.cross(normal, arbitrary)  # first axis in plane
    axis1 /= np.linalg.norm(axis1)
    
    axis2 = np.cross(normal, axis1)  # second axis perpendicular to first and normal
    
    projected = []
    for p in points:
        vec = p - centroid
        x = np.dot(vec, axis1)
        y = np.dot(vec, axis2)
        projected.append([x, y])
    return np.array(projected)

def triangle_sides(coords):
    sides = []
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.linalg.norm(coords[i] - coords[j])
            sides.append(dist)
    return np.array(sides)

def triangle_perimeter(sides):
    return np.sum(sides)

def triangle_area(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * abs(x[0]*y[1] + x[1]*y[2] + x[2]*y[0] - y[0]*x[1] - y[1]*x[2] - y[2]*x[0])

def polar_moment(coords):
    centroid = np.mean(coords, axis=0)
    r2 = np.sum((coords - centroid)**2, axis=1)
    return np.sum(r2)

def side_ratio(sides):
    return np.min(sides) / np.max(sides)

def normal_coefficient(coords):
    peri = triangle_perimeter(triangle_sides(coords))
    area = triangle_area(coords)
    return (4 * np.sqrt(3) * area) / (peri**2)



##STEP 5 - loop over local triads formed from neighbors within FoV using KDTree, calculate properties and store

# properties = []
# max_triads = 5000000  
# seen_triads = set()

# for i, v in enumerate(vectors):
#     neighbor_idxs = tree.query_ball_point(v, r=2 * np.sin(MAX_ANGLE_RAD / 2))  
#     neighbor_idxs = [idx for idx in neighbor_idxs if idx != i]
#     if len(neighbor_idxs) < 2:
#         continue
#     for j, k in itertools.combinations(neighbor_idxs, 2):
#         triad = tuple(sorted([i, j, k]))
#         if triad in seen_triads:
#             continue  
#         seen_triads.add(triad)

#         a, b, c = vectors[list(triad)]
#         ab = angular_distance(a, b)
#         bc = angular_distance(b, c)
#         ca = angular_distance(c, a)

#         if max(ab, bc, ca) > MAX_ANGLE_RAD:
#             continue

#         pts = np.array([a, b, c])
#         proj = project_to_plane(pts)
#         sides = triangle_sides(proj)

#         perimeter = triangle_perimeter(sides)
#         area = triangle_area(proj)
#         pmoment = polar_moment(proj)
#         ratio = side_ratio(sides)
#         norm_coeff = normal_coefficient(proj)

#         properties.append({
#             'triad': triad,
#             'perimeter': perimeter,
#             'area': area,
#             'polar_moment': pmoment,
#             'side_ratio': ratio,
#             'normal_coefficient': norm_coeff
#         })

#         if len(properties) >= max_triads:
#             break
#     if len(properties) >= max_triads:
#         break

# triad_df = pd.DataFrame(properties)

##Loading precomputed catalog triad properties
triad_df = pd.read_csv("triad_properties.csv", nrows = 100000)



##STEP 7 - INCLUDING THE MONTE CARLO SIMULATION - simulates star traids for comparison help your feature stats (mean, covariance, etc.) generalize better to what your star tracker will actually see. 
##Better for distorted/noisy matching - still estimating mean and convergence of simulated triad properties
def monte_carlo_simulation(vectors, num_samples=100000, fov_deg=30):
    simulated_properties = []
    fov_rad = np.deg2rad(fov_deg)
    chord_dist = 2 * np.sin(fov_rad / 2)

    tree = KDTree(vectors)

    for _ in range(num_samples):
        # Randomly choose a center point on the sphere
        rand_dir = np.random.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir)  # Normalize to unit vector

        # Find nearby stars (mock camera FoV)
        neighbor_idxs = tree.query_ball_point(rand_dir, r=chord_dist)
        if len(neighbor_idxs) < 3:
            continue  # Not enough stars in this random FoV

        sample_indices = np.random.choice(neighbor_idxs, 3, replace=False)
        pts = vectors[sample_indices]

        # Project to plane and calculate properties
        proj = project_to_plane(pts)
        sides = triangle_sides(proj)

        perimeter = triangle_perimeter(sides)
        area = triangle_area(proj)
        pmoment = polar_moment(proj)
        ratio = side_ratio(sides)
        norm_coeff = normal_coefficient(proj)

        simulated_properties.append([
            perimeter, area, pmoment, ratio, norm_coeff
        ])

    return np.array(simulated_properties)

#now running the simulation
#simulated_data = monte_carlo_simulation(vectors, num_samples=100000, fov_deg=30)

#computing empirical mean and covariance of the simulated features, invert matrix for multiplication
#mean_vector = np.mean(simulated_data, axis=0)
#cov_matrix = np.cov(simulated_data, rowvar=False)
#inv_cov_matrix = np.linalg.inv(cov_matrix)

#save evrything to avoid having to rerun simulation everytime testing matching
#np.save("montecarlo_mean.npy", mean_vector)
#np.save("montecarlo_cov.npy", cov_matrix)
#np.save("montecarlo_cov_inv.npy", inv_cov_matrix)



##STEP 8 - STARTING THE MATCHING PROCESS, using the concept of Mahalanobis distance to compare triads from the image 
##to those from the star catalog and find the matching traids by applying constraints
import numpy as np
from scipy.spatial import distance

catalog_features = triad_df[['perimeter', 'area', 'polar_moment', 'side_ratio', 'normal_coefficient']].values #extracting values from the filtered catalog

#computing the mean and covariance/standard dev. of the catalog features
mean_vector = np.mean(catalog_features, axis=0)
cov_matrix = np.cov(catalog_features, rowvar=False)

#invert the covariance matrix once so matrix multiplication is possible
inv_cov_matrix = np.linalg.inv(cov_matrix)

def mahalanobis_distance(x, mean, inv_cov): #defining the function for using the mahalanobis distance
    diff = x - mean
    return np.sqrt(diff.T @ inv_cov @ diff)

mean_vector = np.load("montecarlo_mean.npy")
inv_cov_matrix = np.load("montecarlo_cov_inv.npy")



##STEP 9 - creating the matching loop, now have the triads from the catalog, the image and the monte carlo simulation, starting testing
from scipy.spatial import distance

image_triads_df = pd.read_csv("detected-star-properties.csv")

catalog_features = triad_df[['perimeter', 'area', 'polar_moment', 'side_ratio', 'normal_coefficient']].values

matches = []

for idx, image_row in image_triads_df.iterrows():
    image_vector = image_row[['perimeter', 'area', 'polar_moment', 'side_ratio', 'normal_coefficient']].values
    
    # Compute Mahalanobis distances to all catalog triads
    dists = [distance.mahalanobis(image_vector, c, inv_cov_matrix) for c in catalog_features]

    # Get the index of the closest match
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    best_match = triad_df.iloc[min_idx]

    matches.append({
        'image_triad_index': idx,
        'matched_catalog_triad': best_match['triad'],
        'mahalanobis_distance': min_dist
    })

matches_df = pd.DataFrame(matches)
matches_df.to_csv("matched_triads.csv", index=False)
print("Saved matched triads to 'matched_triads.csv'")
