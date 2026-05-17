import cv2
import numpy as np

from .common import safe_crop, to_gray
from .edges import ssim_score, xdog


def find_panel_boxes(image, min_area_ratio=0.01):
    gray = to_gray(image)
    inverted = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    min_area = height * width * min_area_ratio
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= min_area and w > 32 and h > 32:
            boxes.append([x, y, x + w, y + h])

    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def boxes_from_detections(detections):
    if isinstance(detections, dict):
        detections = detections.get('panels', detections.get('panel_boxes', []))

    boxes = []
    for item in detections or []:
        box = item.get('bbox') if isinstance(item, dict) else item
        if box and len(box) == 4:
            boxes.append([int(round(value)) for value in box])
    return boxes


def mask_from_box(image_shape, box):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 255
    return mask


def extract_panel_pair(bw_image, color_image, box):
    bw_crop, clipped = safe_crop(bw_image, box)
    color_crop, _ = safe_crop(color_image, clipped)
    return bw_crop, color_crop, clipped


def apply_panel_mask(panel, mask):
    if mask is None:
        return panel

    if mask.shape[:2] != panel.shape[:2]:
        mask = cv2.resize(mask, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)

    output = panel.copy()
    output[mask == 0] = 255
    return output


def panel_pair_score(bw_panel, color_panel):
    bw_edges = xdog(bw_panel)
    color_edges = xdog(color_panel)
    return ssim_score(bw_edges, color_edges)


class SamPanelSegmenter:
    def __init__(self, checkpoint, model_type='vit_h', device='cuda'):
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def segment_boxes(self, image, boxes):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        masks = []

        for box in boxes:
            mask_candidates, scores, _ = self.predictor.predict(
                box=np.array(box, dtype=np.float32),
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            masks.append(mask_candidates[best_index].astype(np.uint8) * 255)

        return masks
