import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass

def detect_stars(img, sigma_thresh=3.0, min_size=2, max_stars=30):
    """Detect star centroids and intensities in a grayscale image."""
    blurred = gaussian_filter(img, sigma=1.0)
    bg = np.median(blurred)
    thresh = bg + sigma_thresh * np.std(blurred)
    binary = blurred > thresh
    labeled, num_labels = label(binary)
    coms = center_of_mass(blurred, labels=labeled, index=range(1, num_labels + 1))
    sizes = [np.sum(labeled == i) for i in range(1, num_labels + 1)]
    intensities = [np.sum(blurred[labeled == i]) for i in range(1, num_labels + 1)]
    valid_idx = [i for i in range(num_labels) if sizes[i] >= min_size]
    centroids = np.array([coms[i] for i in valid_idx])
    intensities = np.array([intensities[i] for i in valid_idx])
    if len(centroids) > 0:
        sort_idx = np.argsort(-intensities)[:max_stars]
        centroids = centroids[sort_idx]
        intensities = intensities[sort_idx]
    else:
        centroids = np.empty((0, 2))
    return centroids, intensities 