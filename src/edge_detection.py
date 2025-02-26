import cv2
import numpy as np


def canny_edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Applies Canny Edge Detection to a given grayscale and blurred image.

    :param image: Input image (must be a grayscale and blurred image)
    :param low_threshold: Lower threshold for the hysteresis procedure
    :param high_threshold: Upper threshold for the hysteresis procedure
    :return: Image after applying Canny Edge Detection
    """
    if image is None or len(image.shape) != 2:
        raise ValueError("Input image must be a non-empty grayscale image.")

    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def morphological_closing(image, kernel_size: int = 3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return edges_closed
