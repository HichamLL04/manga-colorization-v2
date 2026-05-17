import cv2
import numpy as np

from .common import to_gray


def xdog(image, sigma=0.5, k=4.0, gamma=0.95, epsilon=0.0, phi=90.0):
    gray = to_gray(image).astype(np.float32) / 255.0
    blur_small = cv2.GaussianBlur(gray, (0, 0), sigma)
    blur_large = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = blur_small - gamma * blur_large
    response = np.ones_like(dog)
    mask = dog < epsilon
    response[mask] = 1.0 + np.tanh(phi * (dog[mask] - epsilon))
    response = np.clip(response, 0.0, 1.0)
    _, binary = cv2.threshold((response * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def ssim_score(first, second):
    first = to_gray(first).astype(np.float32)
    second = to_gray(second).astype(np.float32)

    if first.shape != second.shape:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(first, kernel, sigma)
    mu2 = cv2.GaussianBlur(second, kernel, sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(first * first, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(second * second, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(first * second, kernel, sigma) - mu12
    numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(numerator / (denominator + 1e-12)))

