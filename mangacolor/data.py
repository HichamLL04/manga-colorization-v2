"""Aligned training data and synthetic user controls for manga colorization.

Color images are supervision; monochrome images are explicitly unlabelled.
The manifest keeps every synthetic variant of an image in the same split.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _images(directory: Path) -> list[Path]:
    return sorted((p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                  key=lambda p: p.relative_to(directory).as_posix().casefold())


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")


def _is_color(image: Image.Image) -> bool:
    # Ignore compression noise and small channel differences in scanned paper.
    rgb = np.asarray(image, dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    saturation = spread / np.maximum(rgb.max(axis=2), 1)
    return bool(np.mean((spread > 10) & (saturation > .08)) > .001)


def _pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def _lineart(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    # Padding prevents FIND_EDGES' unchanged frame becoming a black border.
    padded = ImageOps.expand(gray, border=2, fill="white")
    edges = padded.filter(ImageFilter.GaussianBlur(.4)).filter(ImageFilter.FIND_EDGES)
    result = ImageOps.invert(edges).point(lambda value: min(255, max(0, (value - 150) * 255 // 105)))
    return result.crop((2, 2, image.width + 2, image.height + 2))


def _save_generated(image: Image.Image, path: Path) -> None:
    if path.exists():
        # Generated names are content-addressed: never silently overwrite data.
        with Image.open(path) as existing:
            if existing.mode == image.mode and existing.size == image.size and existing.tobytes() == image.tobytes():
                return
        raise FileExistsError(f"Existing generated image has different contents: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def prepare_dataset(color_dir, out_dir, bw_dir=None, val_fraction=.1, seed=42) -> Path:
    """Build a deterministic JSONL manifest and PNG pairs without altering sources.

    Optional real BW pairs match the complete relative path without extension;
    different dimensions raise an error. Grayscale sources have no target and
    use the ``unlabeled`` split. Paths in the manifest are relative to it.
    Existing datasets may be regenerated only when the manifest is identical.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    source = Path(color_dir).resolve()
    output = Path(out_dir).resolve()
    paired_root = Path(bw_dir).resolve() if bw_dir is not None else None
    if not source.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {source}")
    for original in (source, paired_root):
        if original is not None and (output == original or original in output.parents):
            raise ValueError("Output must be outside the source image directories")
    if paired_root is not None and not paired_root.is_dir():
        raise FileNotFoundError(f"BW directory does not exist: {paired_root}")

    pairs = {}
    if paired_root is not None:
        for path in _images(paired_root):
            key = path.relative_to(paired_root).with_suffix("").as_posix().casefold()
            if key in pairs:
                raise ValueError(f"Ambiguous BW pair for relative stem {key!r}")
            pairs[key] = path

    # Validate all pair dimensions before writing any generated images.
    records = []
    for path in _images(source):
        image = _read_rgb(path)
        relative = path.relative_to(source).as_posix()
        key = Path(relative).with_suffix("").as_posix().casefold()
        pair = pairs.get(key)
        if pair is not None:
            pair_image = _read_rgb(pair)
            if pair_image.size != image.size:
                raise ValueError(f"Unaligned pair {relative}: color {image.size}, BW {pair_image.size}")
        records.append({"source": path, "relative": relative, "hash": _pixel_hash(image),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "width": image.width, "height": image.height,
                        "color": _is_color(image), "pair": pair,
                        "pair_hash": hashlib.sha256(pair.read_bytes()).hexdigest()[:12] if pair else None})
    if not records:
        raise ValueError(f"No supported images found in {source}")

    # When several works are stored in top-level folders, split whole works.
    # With a single work, fall back to image hashes. Duplicate pixels merge
    # works into one split group even when copied under different names.
    work_names = {Path(r["relative"]).parts[0] for r in records
                  if r["color"] and len(Path(r["relative"]).parts) > 1}
    group_by_work = len(work_names) > 1
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for record in records:
        parts = Path(record["relative"]).parts
        record["source_group"] = parts[0] if len(parts) > 1 else None
        key = "work:" + parts[0] if group_by_work and len(parts) > 1 else "image:" + record["hash"]
        record["split_key"] = key
        if record["color"]:
            union(key, "image:" + record["hash"])
    groups = sorted({find(r["split_key"]) for r in records if r["color"]})
    permutation = np.random.default_rng(seed).permutation(len(groups))
    n_val = min(max(0, len(groups) - 1), int(round(len(groups) * val_fraction)))
    if val_fraction > 0 and len(groups) > 1:
        n_val = min(len(groups) - 1, max(1, n_val))
    validation = {groups[int(i)] for i in permutation[:n_val]}
    manifest_rows = []
    for record in records:
        digest = record["hash"]
        split_group = find(record["split_key"]) if record["color"] else "image:" + digest
        common = {"source": record["relative"], "sha256": record["sha256"],
                  "group": digest, "source_group": record["source_group"], "split_group": split_group,
                  "width": record["width"], "height": record["height"]}
        if not record["color"]:
            relative_input = f"inputs/{digest}-unlabeled.png"
            manifest_rows.append({**common, "id": f"{digest}-unlabeled", "input": relative_input,
                                  "target": None, "split": "unlabeled", "supervision": "unlabeled",
                                  "variant": "unlabeled"})
            continue
        relative_target = f"targets/{digest}.png"
        variants = ["luminance", "lineart"]
        if record["pair"] is not None:
            variants.append(f"paired-{record['pair_hash']}")
        for variant in variants:
            relative_input = f"inputs/{digest}-{variant}.png"
            manifest_rows.append({**common, "id": f"{digest}-{variant}", "input": relative_input,
                                  "target": relative_target, "split": "val" if split_group in validation else "train",
                                  "supervision": "paired" if variant.startswith("paired-") else "synthetic",
                                  "variant": variant})

    # Exact duplicate images should not multiply their training weight.
    unique_rows = {row["id"]: row for row in manifest_rows}
    manifest = output / "manifest.jsonl"
    contents = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                       for row in sorted(unique_rows.values(), key=lambda r: r["id"]))
    if manifest.exists() and manifest.read_bytes() != contents.encode("utf-8"):
        raise FileExistsError(f"Dataset manifest already exists with different data: {manifest}. Choose a new output directory.")

    # Manifest identity is checked before creating any new output images.
    for record in records:
        image = _read_rgb(record["source"])
        digest = record["hash"]
        if not record["color"]:
            _save_generated(ImageOps.grayscale(image), output / f"inputs/{digest}-unlabeled.png")
            continue
        _save_generated(image, output / f"targets/{digest}.png")
        _save_generated(ImageOps.grayscale(image), output / f"inputs/{digest}-luminance.png")
        _save_generated(_lineart(image), output / f"inputs/{digest}-lineart.png")
        if record["pair"] is not None:
            _save_generated(ImageOps.grayscale(_read_rgb(record["pair"])),
                            output / f"inputs/{digest}-paired-{record['pair_hash']}.png")
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".jsonl", dir=output)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
        os.replace(temporary, manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest


def extract_palette(image: Image.Image, k=8) -> np.ndarray:
    """Return frequency-ordered RGB uint8 swatches, padded to exactly k rows."""
    if not isinstance(k, int) or not 1 <= k <= 256:
        raise ValueError("Palette size must be an integer between 1 and 256")
    sample = image.convert("RGB").copy()
    sample.thumbnail((256, 256), Image.Resampling.NEAREST)
    indexed = sample.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    counts = sorted(indexed.getcolors() or [], key=lambda pair: (-pair[0], pair[1]))
    colors = np.asarray(indexed.getpalette(), dtype=np.uint8).reshape(-1, 3)
    result = [colors[index] for _, index in counts]
    if not result:
        raise ValueError("Cannot extract a palette from an empty image")
    result.extend([result[-1]] * (k - len(result)))
    return np.stack(result[:k]).astype(np.uint8)


def _recolor(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    hsv = np.array(image.convert("HSV"), dtype=np.uint8)
    colored = hsv[..., 1] > 20
    hue = (hsv[..., 0].astype(np.int16) + int(rng.integers(0, 256))) % 256
    saturation = np.clip(hsv[..., 1].astype(np.float32) * rng.uniform(.7, 1.3), 0, 255)
    hsv[..., 0] = np.where(colored, hue, hsv[..., 0]).astype(np.uint8)
    hsv[..., 1] = np.where(colored, saturation, hsv[..., 1]).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _hint_mask(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    if rng.random() < .2:
        return mask
    candidates = np.argwhere(rgb.min(axis=2) < 240)
    if len(candidates) == 0:
        candidates = np.argwhere(np.ones((height, width), dtype=bool))
    for _ in range(int(rng.integers(1, 9))):
        y, x = candidates[int(rng.integers(len(candidates)))]
        if rng.random() < .4:
            mask[y, x] = 1
            continue
        # Flood fill is restricted to similar target colors, so regional hints
        # cannot form rectangles that cut across ink or neighboring colors.
        distance = np.linalg.norm(rgb.astype(np.float32) - rgb[y, x].astype(np.float32), axis=2)
        allowed = (distance <= float(rng.uniform(15, 45)))
        connected = Image.fromarray(np.where(allowed, 0, 255).astype(np.uint8), mode="L")
        ImageDraw.floodfill(connected, (int(x), int(y)), 128, thresh=0)
        component = np.asarray(connected) == 128
        if rng.random() < .25:
            shape = np.ones_like(mask, dtype=bool)
        else:
            brush = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(brush)
            radius = int(rng.integers(2, max(3, min(height, width) // 10)))
            if rng.random() < .5:
                draw.ellipse((int(x-radius), int(y-radius), int(x+radius), int(y+radius)), fill=255)
            else:
                dx, dy = rng.integers(-radius * 3, radius * 3 + 1, size=2)
                draw.line((int(x), int(y), int(x+dx), int(y+dy)), fill=255, width=radius * 2 + 1)
            shape = np.asarray(brush) > 0
        mask[component & shape] = 1
    return mask


def _sample_weight(row: dict) -> float:
    try:
        weight = float(row.get("weight", .25 if row.get("supervision") == "pseudo" else 1.))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid sample weight for {row.get('id', '<unknown>')}") from error
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"Invalid sample weight for {row.get('id', '<unknown>')}: {weight}")
    return weight


class PaletteDataset(Dataset):
    """Five-channel input: gray, masked normalized RGB, and hint presence.

    ``set_epoch`` changes deterministic train augmentations each epoch. Validation
    always uses the same crop, hints, and palette for a given seed and index.
    Unlabelled records never participate in supervised training.
    """
    def __init__(self, manifest, split="train", size=256, palette_size=8, seed=42):
        if not isinstance(size, int) or size <= 0 or size % 32:
            raise ValueError("Training size must be a positive multiple of 32")
        if split not in {"train", "val"}:
            raise ValueError("Supervised split must be train or val")
        if not isinstance(palette_size, int) or not 1 <= palette_size <= 256:
            raise ValueError("palette_size must be in [1, 256]")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.size, self.palette_size, self.seed, self.split = size, palette_size, seed, split
        self.epoch = 0
        self.records = []
        split_membership = {}
        with self.manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Manifest line {line_number} must be an object")
                row_split = row.get("split")
                supervised = row.get("target") and row.get("supervision") != "unlabeled"
                if row_split in {"train", "val"} and supervised:
                    if not isinstance(row.get("id"), str) or not row["id"]:
                        raise ValueError(f"Missing id on manifest line {line_number}")
                    if not isinstance(row.get("input"), str) or not row["input"]:
                        raise ValueError(f"Missing input on manifest line {line_number}")
                    if not isinstance(row.get("target"), str):
                        raise ValueError(f"Invalid target on manifest line {line_number}")
                    if row_split == "val" and row.get("supervision") == "pseudo":
                        raise ValueError("Pseudo-labels cannot be used as ground truth in the validation split")
                    _sample_weight(row)
                    # Check explicit grouping and resolved file identities. Do
                    # not use source_group here: one-work datasets deliberately
                    # fall back to image-level splits.
                    keys = [(name, str(row[name])) for name in ("id", "group", "split_group") if row.get(name)]
                    keys += [("input", str((self.root / row["input"]).resolve())),
                             ("target", str((self.root / row["target"]).resolve()))]
                    for key in keys:
                        previous = split_membership.setdefault(key, row_split)
                        if previous != row_split:
                            raise ValueError(f"Train/validation leakage for {key[0]} {key[1]}")
                    if row_split == split:
                        self.records.append(row)

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.records[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if self.split == "train" else 0, index]))
        with Image.open(self.root / row["input"]) as image:
            sketch = image.convert("L")
        target = _read_rgb(self.root / row["target"])
        if sketch.size != target.size:
            raise ValueError(f"Unaligned sample {row['id']}: {sketch.size} != {target.size}")
        if self.split == "val" and not _is_color(target):
            raise ValueError(f"Validation target is monochrome and supplies no color ground truth: {row['id']}")
        ratio = self.size / min(target.size)
        resized = (max(self.size, round(target.width * ratio)), max(self.size, round(target.height * ratio)))
        sketch = sketch.resize(resized, Image.Resampling.BILINEAR)
        target = target.resize(resized, Image.Resampling.BILINEAR)
        if self.split == "train":
            left = int(rng.integers(0, resized[0] - self.size + 1))
            top = int(rng.integers(0, resized[1] - self.size + 1))
        else:
            left, top = (resized[0] - self.size) // 2, (resized[1] - self.size) // 2
        crop = (left, top, left + self.size, top + self.size)
        sketch, target = sketch.crop(crop), target.crop(crop)
        if self.split == "train" and rng.random() < .5:
            sketch, target = ImageOps.mirror(sketch), ImageOps.mirror(target)
        if self.split == "train" and rng.random() < .5:
            target = _recolor(target, rng)
        rgb = np.asarray(target, dtype=np.uint8)
        target_array = rgb.astype(np.float32) / 127.5 - 1
        mask = _hint_mask(rgb, rng)
        palette_bytes = extract_palette(target, self.palette_size)
        palette_mask = np.zeros(self.palette_size, dtype=np.float32)
        seen = set()
        for i, color in enumerate(palette_bytes):
            key = tuple(int(v) for v in color)
            if key not in seen:
                palette_mask[i] = 1
                seen.add(key)
        palette = palette_bytes.astype(np.float32) / 127.5 - 1
        if rng.random() < .2:
            palette_mask.fill(0)
            palette.fill(0)
        gray = np.asarray(sketch, dtype=np.float32) / 255
        inputs = np.concatenate((gray[None], (target_array * mask[..., None]).transpose(2, 0, 1), mask[None]), axis=0)
        weight = _sample_weight(row)
        return {"input": torch.from_numpy(inputs.copy()), "target": torch.from_numpy(target_array.transpose(2, 0, 1).copy()),
                "palette": torch.from_numpy(palette.copy()), "palette_mask": torch.from_numpy(palette_mask),
                "hint_mask": torch.from_numpy(mask[None].copy()), "weight": torch.tensor(weight, dtype=torch.float32), "id": row["id"]}
