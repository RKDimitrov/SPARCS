import cv2
import numpy as np

def detect_stars(image_path, threshold=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {image_path}")
        return [], (0, 0)
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    star_centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            star_centroids.append((cx, cy))
    height, width = img.shape
    return star_centroids, (width, height)

if __name__ == "__main__":
    image_path = "../stellarium/image.png"  # Change this to your image path
    stars, (width, height) = detect_stars(image_path)
    center_x, center_y = width // 2, height // 2
    print(f"Detected {len(stars)} stars (centered at (0,0)):")
    for i, (x, y) in enumerate(stars):
        x_centered = x - center_x
        y_centered = y - center_y
        print(f"Star {i+1}: (x={x_centered}, y={y_centered})")