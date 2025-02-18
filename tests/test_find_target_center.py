import os

import cv2

from src.geometry_correction import find_target_center
from src.image_loader import load_image


def test_find_target_center_ellipse():
    # Get the absolute path to the image
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "..", "data", "target.jpeg")

    # Load and display original image
    image = load_image(img_path)  # Load image with absolute path


    cx, cy = find_target_center(image)  # Make sure it returns (cx, cy)

    # Draw a dot at the detected center
    cv2.circle(image, (int(cx), int(cy)), 25, (0, 0, 255), -1)

    # 4. Save (or display) the result
    output_path = "test_output.jpg"
    cv2.imwrite(output_path, image)
    print(f"Test output saved to: {output_path}")

    cv2.imshow("Center Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
