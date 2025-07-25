import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import itertools
import pandas as pd

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

image_path = "../images/MatchingImage.png"  # Updated path for image
img, img_pil = load_image(image_path)
centroids, intensities = detect_stars(img)

print(f"Detected {len(centroids)} stars")
if len(centroids) > 0:
    print("Centroids (y, x):")
    print(centroids)
    print("Intensities:")
    print(intensities)

# Save images
save_grayscale(img_pil, "../outputs/grayscale_30fov_n2.png")
plot_detections(img, centroids, "../outputs/debug_detected_stars_30fov__n2.png")

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

# Generate all possible triad combinations (triplets of star indices)
def generate_triads(num_stars):
    if num_stars < 3:
        return []
    triad_indices = list(itertools.combinations(range(num_stars), 3))
    return triad_indices

# Generate triads from detected stars
num_stars = len(star_vectors)
triad_indices = generate_triads(num_stars)

print(f"Number of possible triads: {len(triad_indices)}")


# Define helper functions for triad properties
def angular_distance(a, b):
    dot = np.dot(a, b)
    return np.arccos(np.clip(dot, -1.0, 1.0))  # In radians

def project_to_plane(pts):
    # Gnomonic projection approximation for small FOV: project to tangent plane
    # Using z as forward, x/y as plane coords
    proj = pts[:, :2] / pts[:, 2][:, np.newaxis]
    return proj

def triangle_sides(proj):
    side_ab = np.linalg.norm(proj[0] - proj[1])
    side_bc = np.linalg.norm(proj[1] - proj[2])
    side_ca = np.linalg.norm(proj[2] - proj[0])
    return np.array([side_ab, side_bc, side_ca])

def triangle_perimeter(sides):
    return np.sum(sides)

def triangle_area(proj):
    x = proj[:, 0]
    y = proj[:, 1]
    return 0.5 * abs(x[0]*y[1] + x[1]*y[2] + x[2]*y[0] - y[0]*x[1] - y[1]*x[2] - y[2]*x[0])

def polar_moment(proj):
    centroid = np.mean(proj, axis=0)
    r2 = np.sum((proj - centroid)**2, axis=1)
    return np.sum(r2)

def side_ratio(sides):
    return np.min(sides) / np.max(sides)

def normal_coefficient(proj):
    sides = triangle_sides(proj)
    peri = triangle_perimeter(sides)
    area = triangle_area(proj)
    return (4 * np.sqrt(3) * area) / (peri**2)

# Generate properties for each triad
properties = []
MAX_ANGLE_RAD = np.deg2rad(30)  # FOV limit in radians
seen_triads = set()  # To avoid duplicates, though combinations should be unique

for triad_idx in triad_indices:
    triad = tuple(sorted(triad_idx))  # Already sorted from combinations
    if triad in seen_triads:
        continue
    seen_triads.add(triad)

    a, b, c = star_vectors[list(triad)]

    ab = angular_distance(a, b)
    bc = angular_distance(b, c)
    ca = angular_distance(c, a)

    # Skip if max angle exceeds FOV (though unlikely for small N)
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

# Convert to DataFrame
triad_df = pd.DataFrame(properties)
print("Triad Properties DataFrame (head):")
print(triad_df.head())


# Save the triad properties to CSV
triad_df.to_csv(r'.\detected_star_properties_n2.csv', index=False)
print("Triad properties saved to .\triad_properties.csv")


# Create DataFrame with star IDs (indices) and unit vectors
vectors_df = pd.DataFrame({
    #'star_id': range(len(star_vectors)),
    'x': star_vectors[:, 0],
    'y': star_vectors[:, 1],
    'z': star_vectors[:, 2]
})

# Save to CSV
vectors_df.to_csv(r'.\star_vectors.csv', index=False)
print("Detected star vectors saved to .\detected_star_vectors.csv")