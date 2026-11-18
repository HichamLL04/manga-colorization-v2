import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from mangacolor.inference import ColorizationPipeline, hints_tensor, parse_palette, pseudo_label
from mangacolor.training import color_losses


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline():
    if not (ROOT / "v2generator.zip").exists():
        pytest.skip("Supplied v2 checkpoint required")
    torch.set_num_threads(2)
    return ColorizationPipeline(ROOT / "v2generator.zip")


def test_rgba_resize_preserves_premultiplied_color_and_black():
    hints = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
    hints.putpixel((17, 17), (0, 0, 0, 255))
    values = hints_tensor(hints, (32, 32))
    assert values.shape == (4, 32, 32)
    assert float(values[3].sum()) > 0
    torch.testing.assert_close(values[:3], -values[3:].expand(3, -1, -1))


def test_inference_full_dimensions_and_exact_lock(pipeline):
    image = Image.new("RGB", (73, 91), (20, 20, 20))
    hints = Image.new("RGBA", image.size)
    ImageDraw.Draw(hints).rectangle((10, 10, 30, 30), fill=(20, 80, 220, 255))
    result, report = pipeline.colorize(image, hints, size=64, lock_hints=True, preserve_ink=True)
    assert result.size == image.size
    assert result.getpixel((20, 20)) == (20, 80, 220)
    assert report["locked_regions"] is True
    assert report["palette_mode"] == "none"
    with pytest.raises(ValueError, match="medir lo mismo"):
        pipeline.colorize(image, hints.resize((20, 20)), size=64)


def test_native_palette_and_learned_gate(pipeline):
    image = Image.new("RGB", (64, 64), (170, 170, 170))
    result, report = pipeline.colorize(image, palette=parse_palette(["#ef3030", "#3060ef"]), size=64)
    assert result.size == image.size
    assert report["palette_mode"] == "native_sparse_hints"
    with pytest.raises(ValueError, match="v2 nativo"):
        pipeline.colorize(image, palette=parse_palette(["#ef3030"]), size=64, palette_mode="learned")


def test_palette_validation():
    np.testing.assert_array_equal(parse_palette(["#000000", "ffffff"]), [[0, 0, 0], [255, 255, 255]])
    with pytest.raises(ValueError):
        parse_palette(["red"])


def test_masked_hint_loss_is_not_diluted_by_image_area():
    target = torch.zeros(1, 3, 32, 32)
    prediction = target.clone()
    prediction[:, :, 3, 4] = 1
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 3, 4] = 1
    batch = {"target": target, "hint_mask": mask, "palette": torch.zeros(1, 8, 3),
             "palette_mask": torch.zeros(1, 8), "weight": torch.ones(1)}
    losses = color_losses(prediction, target, batch)
    assert float(losses["hint"]) == 1
    assert float(losses["palette"]) == 0
    assert torch.isfinite(losses["total"])
    batch["weight"] = torch.tensor([.25])
    assert float(color_losses(prediction, target, batch)["hint"]) == .25


def test_pseudo_labels_are_marked_and_become_trainable(tmp_path):
    source = tmp_path / "bw.png"
    Image.new("L", (40, 40), 150).save(source)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({"id": "a", "input": "bw.png", "target": None,
                                    "supervision": "unlabeled", "split": "unlabeled"}) + "\n")

    class Teacher:
        metadata = {"teacher": "test"}

        def colorize(self, image, size):
            return Image.new("RGB", image.size, (220, 130, 40)), {}

    destination = pseudo_label(manifest, tmp_path / "pseudo", Teacher())
    row = json.loads(destination.read_text())
    assert row["supervision"] == "pseudo"
    assert row["split"] == "train"
    assert row["weight"] == .25
    from mangacolor.data import PaletteDataset
    assert len(PaletteDataset(destination, size=32)) == 1


def test_pseudo_keeps_reserved_unlabeled_val_out(tmp_path):
    Image.new("L", (40, 40), 150).save(tmp_path / "bw.png")
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({"id": "a", "input": "bw.png", "target": None,
                                    "supervision": "unlabeled", "split": "val"}) + "\n")

    class Teacher:
        metadata = {}

        def colorize(self, image, size):
            raise AssertionError("Reserved unlabeled val must not become supervised")

    destination = pseudo_label(manifest, tmp_path / "pseudo", Teacher())
    assert json.loads(destination.read_text())["target"] is None


def test_pipeline_guided_reconstruction_decoupled_export_size(pipeline):
    image = Image.new("RGB", (73, 91), (200, 200, 200))
    result, report = pipeline.colorize(
        image,
        size=64,
        reconstruct_mode="guided",
        export_size=128,
    )
    assert result.size == (103, 128)
    assert "color_field_key" in report
    assert "reconstruction_key" in report
    assert report["export_dimensions"] == [103, 128]


def test_pipeline_predict_and_reconstruct_api(pipeline):
    image = Image.new("RGB", (64, 64), (180, 180, 180))
    cf = pipeline.predict_color_field(image, size=64)
    assert cf.canonical_size == (64, 64)
    assert cf.rgb_canonical.shape == (64, 64, 3)

    from mangacolor.reconstruction import ReconstructionConfig
    res = pipeline.reconstruct(
        image,
        cf,
        reconstruction_config=ReconstructionConfig(export_size=100),
    )
    assert res.dimensions == (100, 100)
    assert res.image.size == (100, 100)


def test_cli_save_and_reuse_color_field(tmp_path):
    from mangacolor.cli import main
    input_file = tmp_path / "input.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(input_file)
    cf_path = tmp_path / "field.npz"
    out_1 = tmp_path / "out_1.png"

    # Colorize and save field
    main(["colorize", str(input_file), str(out_1), "--size", "64", "--save-color-field", str(cf_path)])
    assert cf_path.exists()
    assert out_1.exists()

    # Reuse field without checkpoint
    out_2 = tmp_path / "out_2.png"
    main(["colorize", str(input_file), str(out_2), "--reuse-color-field", str(cf_path), "--export-size", "128"])
    assert out_2.exists()
    with Image.open(out_2) as img:
        assert img.size == (128, 128)

    # Reconstruct command
    out_3 = tmp_path / "out_3.png"
    main(["reconstruct", str(input_file), str(cf_path), str(out_3), "--export-size", "32"])
    assert out_3.exists()
    with Image.open(out_3) as img:
        assert img.size == (32, 32)

