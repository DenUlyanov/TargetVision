import cv2
import numpy as np


def convert_to_grayscale(image):
    """
    Converts an image to grayscale.

    :param image: The input image (NumPy array).
    :return: Grayscale image.
    """
    if image is None:
        raise ValueError("Invalid image provided for grayscale conversion.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

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


def enhance_contrast(image: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """
    Enhances the contrast of a grayscale image using histogram equalization or CLAHE.

    :param image: Input grayscale image
    :param use_clahe: Whether to use CLAHE (recommended for images with uneven lighting)
    :return: Contrast-enhanced image
    """
    if use_clahe:
        # Use CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
    else:
        # Use simple histogram equalization
        enhanced_image = cv2.equalizeHist(image)

    return enhanced_image
