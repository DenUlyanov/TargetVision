import os
from src.image_processing import load_image, display_image

if __name__ == "__main__":
    # Get the absolute path to the image
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "..", "data", "target.jpeg")

    img = load_image(img_path)  # Load image with absolute path
    display_image(img, "Loaded Image")
