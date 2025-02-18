import cv2


def find_target_center(image):
    edges = cv2.Canny(image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contours found.")

    # Pick the largest contour or the one that is most circular
    contour = max(contours, key=cv2.contourArea)

    # Fit ellipse
    if len(contour) < 5:
        # fitEllipse needs at least 5 points
        raise ValueError("Not enough points to fit an ellipse.")

    ellipse = cv2.fitEllipse(contour)
    (x, y), (major, minor), angle = ellipse

    return x, y


def display_target_center(image, x, y):
    # Draw a red dot at the supposed center
    return cv2.circle(image, (int(x), int(y)), 30, (0, 0, 255), -1)
