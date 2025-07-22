#Detection of stars from the stellarium images

import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from PIL import Image
import matplotlib.pyplot as plt

#Image loading
def load_image(image_path):
    img = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
    return img.astype(np.float32)


#Detect stars
def detect_stars(img, sigma_thresh=3.0, min_size=2, max_stars=20):
    
    blurred = gaussian_filter(img, sigma=1.0)
    bg = np.median(blurred)
    thresh = bg + sigma_thresh * np.std(blurred)
    
    #Binary mask and labeling
    binary = blurred > thresh
    labeled, num_labels = label(binary)
    
    #Centroids, sizes, intensities
    coms = center_of_mass(blurred, labels=labeled, index=range(1, num_labels + 1))
    sizes = [np.sum(labeled == i) for i in range(1, num_labels + 1)]
    intensities = [np.sum(blurred[labeled == i]) for i in range(1, num_labels + 1)]
    
    # Filter by size
    valid_idx = [i for i in range(num_labels) if sizes[i] >= min_size]
    centroids = np.array([coms[i] for i in valid_idx])
    intensities = np.array([intensities[i] for i in valid_idx])
    
    # Sort by intensity descending, take top N
    if len(centroids) > 0:
        sort_idx = np.argsort(-intensities)[:max_stars]
        centroids = centroids[sort_idx]
        intensities = intensities[sort_idx]
    else:
        centroids = np.empty((0, 2))
    
    return centroids, intensities


# Step 6: Debug plot
def plot_detections(img, centroids, save_path='detected_stars.png'):
    plt.imshow(img, cmap='gray')
    if len(centroids) > 0:
        plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=20, label='Detected Stars')  # x, y
    plt.title('Star Detections')
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

# Usage: Replace with your image path
image_path = './stellarium/image2.png'  # Relative path for cross-platform use. Change if your image is elsewhere.
img = load_image(image_path)
centroids, intensities = detect_stars(img)

print(f"Detected {len(centroids)} stars")
if len(centroids) > 0:
    print("Centroids (y, x):")
    print(centroids)
    print("Intensities:")
    print(intensities)

# Visualize
plot_detections(img, centroids)