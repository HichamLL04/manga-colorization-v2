import cv2
import numpy as np

from .common import resize_like, to_gray


def align_pair(bw_image, color_image, min_matches=24, ratio_test=0.75, ransac_threshold=5.0):
    bw_gray = to_gray(bw_image)
    color_gray = to_gray(color_image)
    sift = cv2.SIFT_create()
    kp_bw, desc_bw = sift.detectAndCompute(bw_gray, None)
    kp_color, desc_color = sift.detectAndCompute(color_gray, None)

    if desc_bw is None or desc_color is None:
        return None, {'status': 'no_descriptors', 'inlier_ratio': 0.0, 'matches': 0}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc_color, desc_bw, k=2)
    matches = []

    for first, second in raw_matches:
        if first.distance < ratio_test * second.distance:
            matches.append(first)

    if len(matches) < min_matches:
        return None, {'status': 'too_few_matches', 'inlier_ratio': 0.0, 'matches': len(matches)}

    src = np.float32([kp_color[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_bw[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)

    if matrix is None or inliers is None:
        return None, {'status': 'homography_failed', 'inlier_ratio': 0.0, 'matches': len(matches)}

    inlier_ratio = float(inliers.sum() / len(inliers))
    height, width = bw_image.shape[:2]
    aligned_color = cv2.warpPerspective(color_image, matrix, (width, height), flags=cv2.INTER_LINEAR)

    return aligned_color, {
        'status': 'aligned',
        'inlier_ratio': inlier_ratio,
        'matches': len(matches),
        'inliers': int(inliers.sum()),
        'homography': matrix.tolist(),
    }


def fallback_resize_pair(bw_image, color_image):
    return resize_like(color_image, bw_image), {
        'status': 'resized_without_homography',
        'inlier_ratio': 0.0,
        'matches': 0,
    }

