import os

from src.color_correction import convert_to_grayscale, apply_blur
from src.geometry_correction import find_target_center, apply_geometric_correction, \
    denoise_old_holes, display_target_center
from src.image_loader import load_image, display_image, save_image

if __name__ == "__main__":
    # Get the absolute path to the image
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "..", "data", "target.jpeg")

    # Load and display original image
    image = load_image(img_path)  # Load image with absolute path
    display_image(image, "Loaded Image")

    # Find center of the image
    x, y, r = find_target_center(image)
    print(f"Center: {x},{y}")
    print(f"Radius: {r}")

    # Convert to grayscale
    greyed = convert_to_grayscale(image)
    display_image(greyed, "Grayscale Image")

    # Blur image
    blured = apply_blur(greyed, 53)
    display_image(blured, "Blur image")
    save_image(blured, "before.jpeg")

    # Show center
    # img_centered = display_target_center(blur_img, x, y)
    # display_image(img_centered, "Centered")

    # Apply geometric correction
    geometric = apply_geometric_correction(blured)
    display_image(geometric, "Geometry")

    # Save image for debugging
    save_image(geometric, "after.jpeg")
