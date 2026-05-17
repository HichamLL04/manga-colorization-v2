import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


@dataclass
class PagePair:
    pair_id: str
    title: str
    chapter: str
    page: str
    bw_path: Path
    color_path: Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_image(path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError('could not read image: {}'.format(path))
    return image


def write_image(path, image):
    ensure_dir(Path(path).parent)
    if not cv2.imwrite(str(path), image):
        raise ValueError('could not write image: {}'.format(path))


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def write_json(path, payload):
    ensure_dir(Path(path).parent)
    with Path(path).open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)


def normalize_token(value):
    value = Path(value).stem.lower()
    value = re.sub(r'[^a-z0-9]+', '_', value)
    return re.sub(r'_+', '_', value).strip('_')


def slug(value):
    value = str(value).lower()
    value = re.sub(r'[^a-z0-9]+', '_', value)
    return re.sub(r'_+', '_', value).strip('_') or 'unknown'


def iter_images(root):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def safe_crop(image, box):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def resize_like(image, reference):
    height, width = reference.shape[:2]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def to_gray(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_variance(image):
    return float(np.var(to_gray(image).astype(np.float32)))

