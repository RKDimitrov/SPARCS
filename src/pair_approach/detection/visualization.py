from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def save_grayscale(img_pil, save_path='grayscale.png'):
    """Save a PIL grayscale image to disk."""
    img_pil.save(save_path, quality=95, optimize=True)
    print(f"Grayscale image saved to {save_path}")

def plot_detections(img, centroids, save_path='detected_stars.png'):
    """Plot detected star centroids on the image and save as PNG."""
    plt.imshow(img, cmap='gray')
    if len(centroids) > 0:
        plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=20, label='Detected Stars')
    plt.axis('off')
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close()
    print(f"Debug plot saved to {save_path}") 