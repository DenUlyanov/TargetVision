import os

from skimage.measure import blur_effect

from src.color_correction import convert_to_grayscale, apply_blur
from src.geometry_correction import find_target_center
from src.image_loader import load_image, display_image, save_image

if __name__ == "__main__":

    # Get the absolute path to the image
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "..", "data", "target.jpeg")

    # Load and display original image
    img = load_image(img_path)  # Load image with absolute path
    display_image(img, "Loaded Image")

    # Convert to grayscale
    gray_img = convert_to_grayscale(img)
    display_image(gray_img, "Grayscale Image")

    # Blur image
    blur_img = apply_blur(gray_img, 53)
    display_image(blur_img, "Blur image")

    save_image(gray_img, "latest.jpeg")

    # Find center of the image
    x, y, r = find_target_center(gray_img)
    print(f"Center: {x},{y}")
    print(f"Radius: {r}")
