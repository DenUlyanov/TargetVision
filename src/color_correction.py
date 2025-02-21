import cv2


def convert_to_grayscale(image):
    """
    Converts an image to grayscale.

    :param image: The input image (NumPy array).
    :return: Grayscale image.
    """
    if image is None:
        raise ValueError("Invalid image provided for grayscale conversion.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return binary


def apply_blur(image, kernel_size):
    """
    Converts an image to grayscale.

    :param image: The input image (NumPy array).
    :param kernel_size: Kernel size for blurring
    :return: Blured image.
    """

    if image is None:
        raise ValueError("Invalid image provided for applying blur")

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred
