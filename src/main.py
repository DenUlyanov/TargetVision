import os

from src.color_correction import convert_to_grayscale, apply_blur
from src.geometry_correction import find_target_center, display_target_center
from src.image_loader import load_image, display_image, save_image

if __name__ == "__main__":
    # Get the absolute path to the image
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "..", "data", "target.jpeg")

    # Load and display original image
    base_img = load_image(img_path)  # Load image with absolute path
    display_image(base_img, "Loaded Image")

    # Find center of the image
    x, y = find_target_center(base_img)
    print(f"Center: {x},{y}")

    # Convert to grayscale
    gray_img = convert_to_grayscale(base_img)
    display_image(gray_img, "Grayscale Image")

    # Blur image
    blur_img = apply_blur(gray_img, 53)
    display_image(blur_img, "Blur image")

    # Show center
    img_centered = display_target_center(blur_img, x, y)
    display_image(img_centered, "Centered")

    save_image(gray_img, "latest.jpeg")
