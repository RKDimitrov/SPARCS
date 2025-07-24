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

##STEP 4 - calculating the triads from the catalog but filtering by stars close together in the camera's FoV using KDTree
vectors = hip[['x', 'y', 'z']].values

def angular_distance(v1, v2):
    dot = np.dot(v1, v2)
    return np.arccos(np.clip(dot, -1, 1))

MAX_ANGLE_DEG = 70# Field of View threshold in degrees of camera
MAX_ANGLE_RAD = np.deg2rad(MAX_ANGLE_DEG)

# Build KDTree for fast neighbor search in 3D space
tree = KDTree(vectors)

##STEP 5 - using the triads to find 5 main triangular properties: perimeter, area, polar moment, ratio of shortest to longest side
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

##STEP 6 - loop over local triads formed from neighbors within FoV using KDTree, calculate properties and store
properties = []
max_triads = 50000  # Just starting test
seen_triads = set()

for i, v in enumerate(vectors):
    # Find neighbors within the angular FoV converted to chord distance on unit sphere
    neighbor_idxs = tree.query_ball_point(v, r=2 * np.sin(MAX_ANGLE_RAD / 2))  

    # Remove self index
    neighbor_idxs = [idx for idx in neighbor_idxs if idx != i]

    # Need at least two neighbors to form triads
    if len(neighbor_idxs) < 2:
        continue

    # Form triads from star i and pairs of its neighbors
    for j, k in itertools.combinations(neighbor_idxs, 2):
        triad = tuple(sorted([i, j, k]))
        if triad in seen_triads:
            continue  # avoid duplicate triads
        seen_triads.add(triad)

        a, b, c = vectors[list(triad)]

        ab = angular_distance(a, b)
        bc = angular_distance(b, c)
        ca = angular_distance(c, a)

        # Double check triad is within max angle (should be guaranteed by neighbor search but safe to keep)
        if max(ab, bc, ca) > MAX_ANGLE_RAD:
            continue

        pts = np.array([a, b, c])
        proj = project_to_plane(pts)
        sides = triangle_sides(proj)

        perimeter = triangle_perimeter(sides)
        area = triangle_area(proj)
        pmoment = polar_moment(proj)
        ratio = side_ratio(sides)
        norm_coeff = normal_coefficient(proj)

        properties.append({
            'triad': triad,
            'perimeter': perimeter,
            'area': area,
            'polar_moment': pmoment,
            'side_ratio': ratio,
            'normal_coefficient': norm_coeff
        })

        if len(properties) >= max_triads:
            break
    if len(properties) >= max_triads:
        break

## STEP 7 - Convert to DataFrame

triad_df = pd.DataFrame(properties)
triad_df.to_csv("triads_from_starRecognition.csv", index=False)
print(triad_df.head())


