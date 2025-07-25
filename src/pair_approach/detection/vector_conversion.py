import numpy as np

def centroids_to_vectors(centroids, img_height, img_width, fov_deg):
    """Convert centroids to 3D unit vectors given image size and field of view."""
    fov_rad = np.deg2rad(fov_deg)
    pixel_scale = fov_rad / img_width
    cy, cx = img_height / 2, img_width / 2
    vectors = []
    for y, x in centroids:
        dx = (x - cx) * pixel_scale
        dy = (cy - y) * pixel_scale
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            vec = np.array([0, 0, 1])
        else:
            vec = np.array([dx, dy, 1 - r**2 / 2])
        vectors.append(vec / np.linalg.norm(vec))
    return np.array(vectors) 