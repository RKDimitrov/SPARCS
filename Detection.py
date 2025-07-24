import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Image loading
def load_image(image_path):
    img_pil = Image.open(image_path).convert('L')  # Convert to grayscale
    img = np.array(img_pil).astype(np.float32)  # For processing
    return img, img_pil  # Return numpy for detection, PIL for saving

# Detect stars
def detect_stars(img, sigma_thresh=3.0, min_size=2, max_stars=20):
    blurred = gaussian_filter(img, sigma=1.0)
    bg = np.median(blurred)
    thresh = bg + sigma_thresh * np.std(blurred)
    
    # Binary mask and labeling
    binary = blurred > thresh
    labeled, num_labels = label(binary)
    
    # Centroids, sizes, intensities
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

# Save grayscale image
def save_grayscale(img_pil, save_path='grayscale.png'):
    img_pil.save(save_path, quality=95, optimize=True)  # High quality PNG
    print(f"Grayscale image saved to {save_path}")

# Matplotlib debug plot (optional visualization)
def plot_detections(img, centroids, save_path='detected_stars.png'):
    plt.imshow(img, cmap='gray')
    if len(centroids) > 0:
        plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=20, label='Detected Stars')  # x, y
    #plt.title('Star Detections (Debug)')
    #plt.legend()
    plt.axis('off')
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close()
    print(f"Debug plot saved to {save_path}")

# Usage
image_path = r".\stellarium\30fov3.png"  # Your Stellarium image
img, img_pil = load_image(image_path)
centroids, intensities = detect_stars(img)

print(f"Detected {len(centroids)} stars")
if len(centroids) > 0:
    print("Centroids (y, x):")
    print(centroids)
    print("Intensities:")
    print(intensities)

# Save images
save_grayscale(img_pil, r".\grayscale_30fov.png")
plot_detections(img, centroids, r".\debug_detected_stars_30fov.png")

#Convert to vectors
def centroids_to_vectors(centroids, img_height, img_width, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    pixel_scale = fov_rad / img_width  # Approx radians/pixel (assuming square FOV)
    cy, cx = img_height / 2, img_width / 2  # Image center
    
    vectors = []
    for y, x in centroids:
        dx = (x - cx) * pixel_scale
        dy = (cy - y) * pixel_scale  # y inverted
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            vec = np.array([0, 0, 1])
        else:
            vec = np.array([dx, dy, 1 - r**2 / 2])  # Small-angle approx for unit vector
        vectors.append(vec / np.linalg.norm(vec))
    return np.array(vectors)

# Usage (after detection)
img_height, img_width = img.shape
star_vectors = centroids_to_vectors(centroids, img_height, img_width, fov_deg=30)

print("Star Vectors")
print(star_vectors)