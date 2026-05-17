import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from colorizator import generate_distance_field_map
from dataset_pipeline.common import load_json


def generate_hint_mask(height, width, empty_prob=0.49, full_prob=0.01):
    draw = random.random()
    if draw < full_prob:
        return torch.ones(1, height, width).float()
    if draw < full_prob + empty_prob:
        return torch.zeros(1, height, width).float()

    threshold = min(1.0, max(0.0, abs(random.gauss(1.0, 0.0005))))
    return torch.rand(1, height, width).ge(threshold).float()


def resize_short_side(image, size, interpolation):
    height, width = image.shape[:2]
    if height <= width:
        new_height = size
        new_width = int(round(width * size / height))
    else:
        new_width = size
        new_height = int(round(height * size / width))
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def crop_pair(bw, color, crop_size):
    height, width = bw.shape[:2]
    if height < crop_size or width < crop_size:
        pad_h = max(0, crop_size - height)
        pad_w = max(0, crop_size - width)
        bw = cv2.copyMakeBorder(bw, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        color = cv2.copyMakeBorder(color, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        height, width = bw.shape[:2]

    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return (
        bw[top:top + crop_size, left:left + crop_size],
        color[top:top + crop_size, left:left + crop_size],
    )


class V2AlignedPanelDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split='train',
        crop_size=512,
        input_channels=5,
        augment=True,
        empty_hint_prob=0.49,
        full_hint_prob=0.01,
    ):
        self.dataset_root = Path(dataset_root)
        self.crop_size = crop_size
        self.input_channels = input_channels
        self.augment = augment
        self.empty_hint_prob = empty_hint_prob
        self.full_hint_prob = full_hint_prob
        self.to_tensor = ToTensor()
        metadata = load_json(self.dataset_root / 'metadata.json', default={})
        split_ids = set(metadata.get('splits', {}).get(split, []))
        panels = metadata.get('panels', [])
        self.items = [panel for panel in panels if not split_ids or panel['panel_id'] in split_ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        bw = cv2.imread(str(self.dataset_root / item['bw_path']), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(str(self.dataset_root / item['color_path']), cv2.IMREAD_COLOR)
        if bw is None or color is None:
            raise ValueError('could not read panel pair {}'.format(item['panel_id']))

        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        bw = resize_short_side(bw, self.crop_size, cv2.INTER_AREA)
        color = resize_short_side(color, self.crop_size, cv2.INTER_AREA)
        color = cv2.resize(color, (bw.shape[1], bw.shape[0]), interpolation=cv2.INTER_AREA)
        bw, color = crop_pair(bw, color, self.crop_size)

        if self.augment and random.random() < 0.5:
            bw = np.ascontiguousarray(np.fliplr(bw))
            color = np.ascontiguousarray(np.fliplr(color))

        bw_tensor = self.to_tensor(bw).float()
        color_tensor = self.to_tensor(color).float()
        color_tensor = (color_tensor - 0.5) / 0.5
        hint_mask = generate_hint_mask(
            bw_tensor.shape[1],
            bw_tensor.shape[2],
            empty_prob=self.empty_hint_prob,
            full_prob=self.full_hint_prob,
        )
        hint = torch.cat([color_tensor * hint_mask, hint_mask], 0)

        if self.input_channels == 6:
            dfm = generate_distance_field_map(bw)
            dfm_tensor = torch.from_numpy(dfm).unsqueeze(0).float()
            sketch = torch.cat([bw_tensor, dfm_tensor, hint], 0)
        else:
            sketch = torch.cat([bw_tensor, hint], 0)

        return {
            'sketch': sketch,
            'color': color_tensor,
            'panel_id': item['panel_id'],
        }

