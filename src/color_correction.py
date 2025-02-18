import cv2


def convert_to_grayscale(image):
    """
    Converts an image to grayscale.

    :param image: The input image (NumPy array).
    :return: Grayscale image.
    """
    if image is None:
        raise ValueError("Invalid image provided for grayscale conversion.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
