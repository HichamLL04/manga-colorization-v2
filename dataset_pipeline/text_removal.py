from .common import image_variance, safe_crop


def load_detections_for_path(detections, path):
    if not detections:
        return []

    path = str(path)
    keys = [path]
    if '/' in path:
        keys.append(path.split('/')[-1])

    for key in keys:
        if key in detections:
            value = detections[key]
            if isinstance(value, dict):
                return value.get('texts', value.get('text', []))
            return value

    return []


def remove_text_boxes(image, boxes, variance_threshold=300.0, margin=4):
    cleaned = image.copy()
    removed = []

    for box in boxes:
        if isinstance(box, dict):
            box = box.get('bbox') or box.get('box')
        if not box or len(box) != 4:
            continue

        crop, clipped = safe_crop(cleaned, box)
        x1, y1, x2, y2 = clipped
        height, width = cleaned.shape[:2]
        bx1 = max(0, x1 - margin)
        by1 = max(0, y1 - margin)
        bx2 = min(width, x2 + margin)
        by2 = min(height, y2 + margin)
        context = cleaned[by1:by2, bx1:bx2]

        if image_variance(context) > variance_threshold:
            continue

        fill_color = context.reshape(-1, context.shape[-1]).mean(axis=0)
        cleaned[y1:y2, x1:x2] = fill_color
        removed.append([x1, y1, x2, y2])

    return cleaned, removed


class MagiDetector:
    """Thin optional adapter.

    The MAGI Python API is not part of this repository, so the pipeline primarily
    consumes detector JSON. If a local MAGI wrapper exposes a compatible
    `predict_text_boxes(image)` method, pass it here and the same cleaning logic
    can be reused.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, image):
        return self.model.predict_text_boxes(image)
