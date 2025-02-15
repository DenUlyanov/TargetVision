import os

import numpy as np
import pytest

from src.image_processing import load_image


def test_load_image_valid():
    """Test loading a valid image."""
    test_image_path = "data/target.jpeg"
    assert os.path.exists(test_image_path), "Test image does not exist"
    img = load_image(test_image_path)
    assert img is not None
    assert isinstance(img, np.ndarray)  # Ensure it's a valid OpenCV image


def test_load_image_invalid():
    """Test loading an invalid image."""
    with pytest.raises(FileNotFoundError):
        load_image("data/non_existent_image.jpg")
