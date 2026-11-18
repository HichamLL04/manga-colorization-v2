"""Reproducible mechanical verification; generated shapes are not a quality benchmark.

Run from the project root: python scripts/verify_pipeline.py
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from mangacolor.checkpoints import export_native, load_model
from mangacolor.data import prepare_dataset
from mangacolor.inference import ColorizationPipeline, parse_palette
from mangacolor.model import NativeGenerator
from mangacolor.training import train


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    output = root / "output" / "pipeline_check"
    source = output / "sources"
    source.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        image = Image.new("RGB", (96, 128), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, 65, 66), fill=(200 - i * 9, 80 + i * 11, 100), outline="black", width=3)
        draw.rectangle((24, 74, 87, 116), fill=(40 + i * 9, 90, 200 - i * 5), outline="black", width=3)
        image.save(source / f"shape_{i:02}.png")
    manifest = prepare_dataset(source, output / "dataset", val_fraction=.2)
    args = SimpleNamespace(checkpoint=str(root / "v2generator.zip"), manifest=str(manifest),
                           output=str(output / "training"), resume=None, device="cpu", seed=42,
                           scope="decoder", lr=2e-5, size=64, batch_size=1, workers=0,
                           epochs=1, max_steps=3, validation_batches=2)
    if (Path(args.output) / "last.pt").exists():
        raise FileExistsError("Use a fresh output/pipeline_check for a new verification run.")
    checkpoint = train(args)
    model, metadata = load_model(checkpoint)
    assert metadata["palette_trained"] and metadata["palette_updates"] > 0
    exported = output / "native_v2.pt"
    export_native(exported, model)
    NativeGenerator().load_state_dict(torch.load(exported, weights_only=True, map_location="cpu"), strict=True)
    pipeline = ColorizationPipeline(checkpoint)
    image = Image.open(source / "shape_00.png").convert("L").convert("RGB")
    red, _ = pipeline.colorize(image, palette=parse_palette(["#ec3030"]), size=64, palette_mode="learned")
    blue, _ = pipeline.colorize(image, palette=parse_palette(["#3050ec"]), size=64, palette_mode="learned")
    difference = float(np.abs(np.asarray(red, dtype=float) - np.asarray(blue, dtype=float)).mean())
    red.save(output / "learned_red_smoke.png")
    blue.save(output / "learned_blue_smoke.png")
    assert difference > 0, "Palette adapter must affect the actual output after optimization"
    summary = {"purpose": "mechanical verification only; not artistic quality",
               "steps": metadata["step"], "palette_updates": metadata["palette_updates"],
               "validation": metadata["validation"], "palette_change_mae_0_255": difference,
               "native_export_strict": True, "checkpoint": str(checkpoint)}
    (output / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
