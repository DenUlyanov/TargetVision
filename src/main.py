import os

from src.color_correction import convert_to_grayscale, apply_blur, enhance_contrast
from src.edge_detection import canny_edge_detection, morphological_closing, canny_circle_detection, draw_circles
from src.fourier_transform import fourier_transformation
from src.geometry_correction import find_target_center, apply_geometric_correction
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
    display_image(greyed, "Grayscale")

    # Enhance contrast
    contrasted = enhance_contrast(greyed)
    display_image(contrasted, "Contrasted")

    # Blur image
    blured = apply_blur(contrasted, 3)
    display_image(blured, "Blur image")

    # Show center
    # img_centered = display_target_center(blured, x, y)
    # display_image(img_centered, "Centered")

    # Apply geometric correction
    geometric = apply_geometric_correction(blured)
    display_image(geometric, "Geometry")

    # Apply FFT transformation
    fft_transformed = fourier_transformation(geometric, 10)
    display_image(fft_transformed, "FFT")

    # Apply Canny edge detection
    edged = canny_edge_detection(fft_transformed, 450, 500)
    display_image(edged, "Edged")
    save_image(edged, "edged.jpeg")

    # Apply morphological closing
    cleaned = morphological_closing(edged)
    display_image(cleaned, "Cleaned")
    save_image(cleaned, "cleaned.jpeg")

    # Detect circles
    circles = canny_circle_detection(cleaned)
    circles_visualized = draw_circles(geometric, circles)
    display_image(circles_visualized, "Circles")

    # Save image for debugging
    save_image(circles_visualized, "latest.jpeg")
