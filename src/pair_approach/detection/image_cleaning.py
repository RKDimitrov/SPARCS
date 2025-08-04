## STAR TRACKER IMAGE DETECTION - 3 STEPS INCLUDE NOISE REDUCTION, REMOVING BLURRING AND IDENTIFYING FALSE POSITIVES
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

def denoise_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def deblur_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_image(path):
    original = load_image(path)
    denoised = denoise_image(original)
    sharpened = deblur_image(denoised)
    gray = convert_to_grayscale(sharpened)
    enhanced = enhance_contrast(gray)
    return original, gray, enhanced

def detect_top_brightest(image_gray, num_stars=10, min_distance=10):
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    coordinates = np.column_stack(np.unravel_index(np.argsort(blurred.ravel())[::-1], image_gray.shape))
    
    keypoints = []
    taken = np.zeros_like(image_gray, dtype=np.uint8)
    
    for y, x in coordinates:
        if taken[y, x] != 0:
            continue
        kp = cv2.KeyPoint(float(x), float(y), size=10)
        keypoints.append(kp)
        cv2.circle(taken, (x, y), min_distance, 255, -1)
        if len(keypoints) >= num_stars:
            break

    return keypoints

def check_circularity(patch):
    _, thresh = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity

import scipy.stats

def is_false_positive(image_gray, kp, drop_thresh=25, gradient_ratio=0.65, min_variance=20, circularity_thresh=0.35, edge_density_thresh=300, min_size=5, min_brightness=80):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    r = max(3, int(kp.size // 2))

    patch = image_gray[max(0, y - r): y + r + 1,
                       max(0, x - r): x + r + 1]

    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return True

    center_val = image_gray[y, x]
    if center_val < min_brightness:
        # Too dim to be star
        return True

    mean_surround = np.mean(patch)
    variance = np.var(patch)

    drop = center_val - mean_surround
    drop_ratio = mean_surround / (center_val + 1e-6)

    # Radial profile calculation
    center = (r, r)
    distances = []
    values = []

    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            dist = np.hypot(i - center[0], j - center[1])
            distances.append(dist)
            values.append(patch[i, j])
    distances = np.array(distances)
    values = np.array(values)

    bins = np.digitize(distances, bins=np.arange(0, r + 1))
    radial_profile = [np.mean(values[bins == i]) for i in range(1, r)]

    # Use slope of radial profile for smooth decreasing check
    if len(radial_profile) > 1:
        slope, _, _, _, _ = scipy.stats.linregress(range(len(radial_profile)), radial_profile)
    else:
        slope = 0

    circularity = check_circularity(patch)

    # Edge density (only edges stronger than threshold)
    sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    strong_edges = magnitude[magnitude > 20]
    edge_density = np.mean(strong_edges) if len(strong_edges) > 0 else 0

    reasons = []
    if drop < drop_thresh:
        reasons.append(f"low drop ({drop:.1f})")
    if drop_ratio > gradient_ratio:
        reasons.append(f"high ratio ({drop_ratio:.2f})")
    if variance < min_variance:
        reasons.append(f"low variance ({variance:.1f})")
    if slope > -0.05:  # expect slightly negative slope
        reasons.append(f"radial profile slope {slope:.2f} not decreasing enough")
    if circularity < circularity_thresh:
        reasons.append(f"low circularity ({circularity:.2f})")
    if edge_density > edge_density_thresh:
        reasons.append(f"high edge density ({edge_density:.1f})")
    if kp.size < min_size:
        reasons.append(f"too small ({kp.size:.1f})")

    if reasons:
        print(f"❌ Rejected ({x},{y}) — Reasons: {reasons}")
        return True

    print(f"✅ Accepted ({x},{y}) — Drop: {drop:.1f}, Ratio: {drop_ratio:.2f}, Var: {variance:.1f}, Slope: {slope:.2f}, Circ: {circularity:.2f}, Edges: {edge_density:.2f}, Size: {kp.size:.1f}")
    return False


def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, np.array([]), (0,255,0),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Example usage
if __name__ == "__main__":
    image_path = "drawing.png"  # Replace with your image
    original, gray, enhanced = preprocess_image(image_path)

    # Save intermediate images
    cv2.imwrite("gray_image.png", gray)
    cv2.imwrite("enhanced_image.png", enhanced)

    # Display preprocessing results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(original), plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
    plt.subplot(1, 3, 3), plt.imshow(enhanced, cmap='gray'), plt.title('Contrast Enhanced')
    plt.show()

    # Detect stars (brightest and biggest)
    keypoints = detect_top_brightest(enhanced, num_stars=20, min_distance=10)

    # TEMP: Show raw detections before filtering
    raw_stars = draw_keypoints(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), keypoints)
    plt.imshow(cv2.cvtColor(raw_stars, cv2.COLOR_BGR2RGB))
    plt.title(f"Raw Detected Points: {len(keypoints)} (pre-filtering)")
    plt.axis('off')
    plt.show()

    # Filter out false positives (e.g., drawn dots)
    filtered_keypoints = []
    for kp in keypoints:
        if not is_false_positive(enhanced, kp):
            filtered_keypoints.append(kp)

    # Draw detected stars
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    image_with_stars = draw_keypoints(enhanced_bgr, filtered_keypoints)

    plt.imshow(cv2.cvtColor(image_with_stars, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Real Stars: {len(filtered_keypoints)}")
    plt.axis('off')
    plt.show()


















