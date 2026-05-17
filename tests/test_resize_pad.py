import numpy as np
import pytest

pytest.importorskip('cv2')

from utils.utils import resize_pad


def test_resize_pad_returns_multiples_of_32_for_portrait():
    image = np.zeros((900, 450, 3), dtype=np.float32)

    resized, _ = resize_pad(image, 384)

    assert resized.shape[0] % 32 == 0
    assert resized.shape[1] % 32 == 0


def test_resize_pad_returns_multiples_of_32_for_landscape():
    image = np.zeros((450, 900, 3), dtype=np.float32)

    resized, _ = resize_pad(image, 384)

    assert resized.shape[0] % 32 == 0
    assert resized.shape[1] % 32 == 0


def test_resize_pad_does_not_add_padding_when_already_multiple():
    image = np.zeros((768, 384, 3), dtype=np.float32)

    resized, pad = resize_pad(image, 384)

    assert resized.shape[:2] == (768, 384)
    assert pad == (0, 0)


def test_colorize_crops_padding():
    torch = pytest.importorskip('torch')
    from colorizator import MangaColorizator

    class DummyColorizer:
        def __call__(self, inputs):
            batch, _, height, width = inputs.shape
            return torch.zeros(batch, 3, height, width), None

    colorizator = MangaColorizator.__new__(MangaColorizator)
    colorizator.colorizer = DummyColorizer()
    colorizator.current_image = torch.zeros(1, 1, 64, 96)
    colorizator.current_hint = torch.zeros(1, 4, 64, 96)
    colorizator.current_pad = (10, 20)

    result = colorizator.colorize()

    assert result.shape == (54, 76, 3)
