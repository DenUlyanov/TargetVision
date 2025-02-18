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


def save_image(image, filename, output_dir="output"):
    """
     Saves an image to the specified directory in the project root as a JPEG file.

     :param image: The processed image (NumPy array).
     :param filename: The name of the output file.
     :param output_dir: The directory where the file will be saved.
     """
    # Get absolute path of the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up from `src/`
    output_path = os.path.join(project_root, output_dir)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Construct full path
    file_path = os.path.join(output_path, filename)

    # Save the image as a JPEG
    success = cv2.imwrite(file_path, image)

    if success:
        print(f"Image saved successfully: {file_path}")
    else:
        print(f"Failed to save image: {file_path}")
