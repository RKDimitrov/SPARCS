import numpy as np
import matplotlib.pyplot as plt

def centroids_to_vectors(centroids, img_height, img_width, fov_deg):
    """Convert centroids to 3D unit vectors given image size and field of view."""
    fov_rad = np.deg2rad(fov_deg)
    pixel_scale = fov_rad / img_width
    cy, cx = img_height / 2, img_width / 2
    vectors = []
    for y, x in centroids:
       # angular offsets in radians
       ang_x = (x - cx) * pixel_scale
       ang_y = (cy - y) * pixel_scale
       # pinhole model: direction = [tan(theta_x), tan(theta_y), 1]
       vec = np.array([np.tan(ang_x),
                       np.tan(ang_y),
                       1.0])
       vectors.append(vec / np.linalg.norm(vec))
    return np.array(vectors) 
