import cv2
import numpy as np


def find_target_center(image):
    """Find the center of the target using Hough Circle Transform."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=50, maxRadius=300
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]  # Take the first detected circle
        return (x, y, r)
    else:
        raise ValueError("No target detected in the image.")
