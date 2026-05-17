from pathlib import Path

import cv2
from torch.utils.data import Dataset

from .common import load_json


class PairedMangaPanelDataset(Dataset):
    def __init__(self, dataset_root, split='train', transform=None):
        self.dataset_root = Path(dataset_root)
        self.transform = transform
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
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        sample = {
            'bw': bw,
            'color': color,
            'metadata': item,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

