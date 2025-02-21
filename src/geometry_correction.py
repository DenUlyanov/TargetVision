import cv2
import numpy as np


def find_target_center(image):
    """Find the center and radius of the target using edge detection and ellipse fitting."""
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contours found.")

    # Pick the largest contour based on area
    contour = max(contours, key=cv2.contourArea)

    # Fit ellipse (requires at least 5 points)
    if len(contour) < 5:
        raise ValueError("Not enough points to fit an ellipse.")

    ellipse = cv2.fitEllipse(contour)
    (x, y), (major, minor), angle = ellipse  # Extract parameters

    # Decide if it's a circle or an ellipse
    aspect_ratio = major / minor
    if 0.9 <= aspect_ratio <= 1.1:
        # It's close to a circle
        radius = (major + minor) / 4
    else:
        # It's more elliptical
        radius = min(major, minor) / 2  # Use the smaller axis as a safer radius

    return int(x), int(y), int(radius)


def display_target_center(image, x, y):
    # Draw a red dot at the supposed center
    return cv2.circle(image, (int(x), int(y)), 30, (0, 0, 255), -1)


def denoise_old_holes(image, center, radius, method="morphological_closing"):
    """Remove old shot holes using denoising techniques, focusing on the target area."""

    # Adaptive threshold to detect holes
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    mask = np.zeros_like(thresh)
    cv2.circle(mask, (center[0], center[1]), radius, 255, thickness=-1)
    thresh = cv2.bitwise_and(thresh, mask)

    if method == "morphological_closing":
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    elif method == "median_blur":
        cleaned = cv2.medianBlur(thresh, 5)
    else:
        raise ValueError("Unsupported denoising method")

    return cleaned


def apply_geometric_correction(image):
    """Fix geometric distortion using Perspective Transform based on circular/elliptical detection."""

    # Use adaptive threshold to improve contour detection
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to remove noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    if contours:
        # Select the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit an ellipse if enough points are detected
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (major_axis, minor_axis), angle = ellipse

            # Define the region based on the fitted ellipse
            width = int(major_axis * 1.2)  # Expand slightly
            height = int(minor_axis * 1.2)
            src_pts = np.float32([
                [cx - width // 2, cy - height // 2],
                [cx + width // 2, cy - height // 2],
                [cx + width // 2, cy + height // 2],
                [cx - width // 2, cy + height // 2]
            ])

            output_size = (2000, 2000)
            dst_pts = np.float32([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]])

            # Compute perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            corrected = cv2.warpPerspective(image, M, output_size)
            return corrected

    raise ValueError("Could not detect a valid elliptical target for perspective correction.")

