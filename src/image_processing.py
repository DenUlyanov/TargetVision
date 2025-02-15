import os
import cv2


def load_image(image_path: str):
    """Loads an image from the given path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def display_image(image, window_name="Image", delay=10000):
    """
    Displays an image in a window for a given duration.

    :param image: The image to be displayed.
    :param window_name: Name of the display window.
    :param delay: Time in milliseconds to display the image (default 10 sec).
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()