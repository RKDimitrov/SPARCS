import numpy as np
from PIL import Image

def load_image(image_path):
    """Load an image and return both numpy array (float32, grayscale) and PIL image."""
    img_pil = Image.open(image_path).convert('L')
    img = np.array(img_pil).astype(np.float32)
    return img, img_pil 