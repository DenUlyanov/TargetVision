import cv2
import numpy as np


def fourier_transformation(image, filter_size=30):
    # Apply FFT
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)  # Center the frequency components

    # Apply a simple filter (optional: low pass or high pass)
    crow, ccol = image.shape[0] // 2, image.shape[1] // 2
    fft_shifted[crow - filter_size:crow + filter_size, ccol - filter_size:ccol + filter_size] = 0  # Example high pass filter

    # Apply IFFT to reconstruct the image
    ifft_shifted = np.fft.ifftshift(fft_shifted)
    reconstructed_image = np.fft.ifft2(ifft_shifted)
    final_image = np.abs(reconstructed_image)

    # Normalize and convert to uint8 format
    final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)
    final_image = final_image.astype(np.uint8)

    return final_image
