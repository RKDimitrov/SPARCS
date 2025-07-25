import numpy as np

def pixel_to_unit_vectors(centroids, image_shape, focal_length_px):
    h, w = image_shape
    cx, cy = w / 2, h / 2
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