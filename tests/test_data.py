import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

from mangacolor.data import PaletteDataset, extract_palette, prepare_dataset


def _color(path, color=(200, 50, 20), size=(64, 96)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _rows(manifest):
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]


def test_prepare_deterministic_splits_and_duplicate_groups(tmp_path):
    source = tmp_path / "source"
    for index in range(6):
        _color(source / "book" / f"{index}.png", (30 + index * 20, 100, 220))
    _color(source / "duplicate.png", (30, 100, 220))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.png")}
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=.3, seed=19)
    first = manifest.read_bytes()
    assert prepare_dataset(source, tmp_path / "prepared", val_fraction=.3, seed=19).read_bytes() == first
    rows = _rows(manifest)
    assert len(rows) == 12
    train = {r["group"] for r in rows if r["split"] == "train"}
    val = {r["group"] for r in rows if r["split"] == "val"}
    assert train and val and train.isdisjoint(val)
    assert all(not Path(r["input"]).is_absolute() for r in rows)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.png")}


def test_grayscale_is_unlabeled_not_fake_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (40, 40), (100, 101, 100)).save(source / "gray.png")
    _color(source / "color.png")
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=0)
    unlabeled = [r for r in _rows(manifest) if r["supervision"] == "unlabeled"]
    assert len(unlabeled) == 1
    assert unlabeled[0]["target"] is None
    assert unlabeled[0]["split"] == "unlabeled"
    assert len(PaletteDataset(manifest, size=32)) == 2


def test_pair_uses_relative_stem_and_rejects_misalignment(tmp_path):
    source, bw = tmp_path / "source", tmp_path / "bw"
    _color(source / "book" / "page.png")
    (bw / "book").mkdir(parents=True)
    Image.new("L", (64, 96), 111).save(bw / "book" / "page.bmp")
    _color(bw / "other" / "page.png", size=(12, 12))
    rows = _rows(prepare_dataset(source, tmp_path / "good", bw_dir=bw))
    assert len([r for r in rows if r["supervision"] == "paired"]) == 1
    Image.new("L", (65, 96), 111).save(bw / "book" / "page.bmp")
    with pytest.raises(ValueError, match="Unaligned pair"):
        prepare_dataset(source, tmp_path / "bad", bw_dir=bw)
    assert not (tmp_path / "bad").exists()


def test_palette_and_dataset_contract_seed_and_epoch(tmp_path):
    source = tmp_path / "source"
    _color(source / "one.png")
    _color(source / "two.png", (20, 60, 210))
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=.5)
    ds = PaletteDataset(manifest, size=32, seed=123)
    same = PaletteDataset(manifest, size=32, seed=123)
    first, second = ds[0], same[0]
    assert first["input"].shape == (5, 32, 32)
    assert first["target"].shape == (3, 32, 32)
    assert first["palette"].shape == (8, 3)
    assert first["palette_mask"].shape == (8,)
    assert first["hint_mask"].shape == (1, 32, 32)
    for key in ("input", "target", "palette", "palette_mask", "hint_mask"):
        assert torch.equal(first[key], second[key])
        assert first[key].dtype == torch.float32
    assert torch.equal(first["input"][4:], first["hint_mask"])
    assert torch.allclose(first["input"][1:4], first["target"] * first["hint_mask"])
    assert first["input"][0].min() >= 0 and first["input"][0].max() <= 1
    changed = False
    for epoch in range(1, 10):
        ds.set_epoch(epoch)
        changed |= not torch.equal(ds[0]["input"], first["input"])
    assert changed
    val = PaletteDataset(manifest, split="val", size=32, seed=123)
    expected = val[0]
    val.set_epoch(10)
    assert torch.equal(val[0]["input"], expected["input"])
    palette = extract_palette(Image.new("RGB", (10, 10), (20, 60, 210)), 8)
    assert palette.shape == (8, 3) and palette.dtype == np.uint8
    assert np.all(palette == np.array([20, 60, 210]))


def test_zero_hints_and_black_have_different_presence(tmp_path):
    source = tmp_path / "source"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 32:] = (200, 40, 20)
    source.mkdir()
    Image.fromarray(image).save(source / "page.png")
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=0)
    empty = black = False
    for seed in range(80):
        sample = PaletteDataset(manifest, size=32, seed=seed)[0]
        mask = sample["hint_mask"]
        if not mask.any():
            empty = True
            assert not sample["input"][1:4].any()
        black_pixels = (sample["target"] == -1).all(dim=0)
        chosen = black_pixels & (mask[0] > 0)
        if chosen.any():
            black = True
            assert torch.all(sample["input"][1:4, chosen] == -1)
        if empty and black:
            break
    assert empty and black


def test_output_in_source_and_invalid_training_size_rejected(tmp_path):
    source = tmp_path / "source"
    _color(source / "page.png")
    with pytest.raises(ValueError, match="outside"):
        prepare_dataset(source, source / "generated")
    manifest = prepare_dataset(source, tmp_path / "prepared")
    with pytest.raises(ValueError, match="multiple of 32"):
        PaletteDataset(manifest, size=33)


def test_works_stay_together_and_cross_work_duplicates_cannot_leak(tmp_path):
    source = tmp_path / "source"
    for book in range(3):
        for page in range(2):
            _color(source / f"book{book}" / f"page{page}.png", (30 + book * 30 + page * 5, 100, 200))
    # This duplicate links book0 and book1 into one split component.
    _color(source / "book1" / "duplicate.png", (30, 100, 200))
    rows = _rows(prepare_dataset(source, tmp_path / "prepared", val_fraction=.5))
    for book in range(3):
        assert len({r["split"] for r in rows if r["source_group"] == f"book{book}"}) == 1
    book0 = next(r["split"] for r in rows if r["source_group"] == "book0")
    book1 = next(r["split"] for r in rows if r["source_group"] == "book1")
    assert book0 == book1
    assert {r["split"] for r in rows} == {"train", "val"}


def test_single_image_keeps_training_sample_even_with_large_val_fraction(tmp_path):
    source = tmp_path / "source"
    _color(source / "page.png")
    rows = _rows(prepare_dataset(source, tmp_path / "prepared", val_fraction=.9))
    assert {row["split"] for row in rows} == {"train"}


def test_validation_geometric_transforms_keep_pair_alignment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rgb = np.empty((111, 67, 3), dtype=np.uint8)
    rgb[:] = (230, 70, 15)
    rgb[20:90, 12:45] = (15, 50, 220)
    Image.fromarray(rgb).save(source / "page.png")
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=0)
    rows = _rows(manifest)
    luminance = next(row for row in rows if row["variant"] == "luminance")
    luminance["split"] = "val"
    manifest.write_text(json.dumps(luminance) + "\n", encoding="utf-8")
    sample = PaletteDataset(manifest, split="val", size=32)[0]
    target = ((sample["target"].numpy().transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)
    expected = np.asarray(Image.fromarray(target).convert("L"), dtype=np.float32) / 255
    assert np.max(np.abs(expected - sample["input"][0].numpy())) < 2 / 255


def test_preparation_refuses_different_manifest_without_modifying_dataset(tmp_path):
    source = tmp_path / "source"
    _color(source / "page.png")
    output = tmp_path / "prepared"
    manifest = prepare_dataset(source, output)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    _color(source / "added.png", (10, 70, 240))
    with pytest.raises(FileExistsError, match="manifest already exists with different data"):
        prepare_dataset(source, output)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert manifest.read_bytes() == before[Path("manifest.jsonl")]


def test_pseudo_weights_are_explicit_and_invalid_weights_fail_early(tmp_path):
    source = tmp_path / "source"
    _color(source / "page.png")
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=0)
    row = _rows(manifest)[0]
    row["supervision"] = "pseudo"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert PaletteDataset(manifest, size=32)[0]["weight"].item() == .25
    row["weight"] = .125
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert PaletteDataset(manifest, size=32)[0]["weight"].item() == .125
    for invalid in (0, -1, float("nan"), float("inf"), None, "bad"):
        row["weight"] = invalid
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid sample weight"):
            PaletteDataset(manifest, size=32)


def test_manifest_leakage_and_pseudo_validation_rejected(tmp_path):
    source = tmp_path / "source"
    _color(source / "page.png")
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=0)
    rows = _rows(manifest)
    rows[1]["split"] = "val"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        PaletteDataset(manifest, size=32)
    rows = [rows[1]]
    rows[0]["supervision"] = "pseudo"
    manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Pseudo-labels"):
        PaletteDataset(manifest, split="val", size=32)


def test_monochrome_validation_target_rejected_but_monochrome_crop_allowed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rgb = np.full((128, 32, 3), 255, dtype=np.uint8)
    rgb[:10] = (200, 40, 20)  # Entire centered crop will be white, a valid crop.
    Image.fromarray(rgb).save(source / "page.png")
    manifest = prepare_dataset(source, tmp_path / "prepared", val_fraction=0)
    row = _rows(manifest)[0]
    row["split"] = "val"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert torch.all(PaletteDataset(manifest, split="val", size=32)[0]["target"] == 1)
    Image.new("RGB", (32, 128), "white").save(manifest.parent / row["target"])
    with pytest.raises(ValueError, match="Validation target is monochrome"):
        PaletteDataset(manifest, split="val", size=32)[0]
