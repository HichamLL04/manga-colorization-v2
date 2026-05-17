# WACV 2025 Paired Manga Dataset Pipeline

This branch adds a reproducible local pipeline for building aligned black-and-white/color manga pairs for training your own colorization model. It follows the practical stages described in Golyadkin et al. 2025: text cleaning, page alignment, panel extraction/filtering, captions, and train/test splits.

## Expected Raw Layout

Use fan-scanlation data only when reuse is allowed by the source. Keep excluded titles out of the raw tree.

```text
raw/
  bw/<title>/<chapter>/<page>.png
  color/<title>/<chapter>/<page>.png
```

Page pairing starts with normalized path/name matching. Bad matches are filtered later by SIFT/RANSAC and panel SSIM.

## Optional MAGI Detections

MAGI detects text and panels. Generate detections for BW pages, color pages, or already prepared pages:

```bash
python build_dataset.py detect-magi \
  --image-root raw/bw \
  --output work/magi_bw.json \
  --model ragavsachdeva/magi \
  --device cuda
```

MAGI output is used as JSON so you can run detection once and resume the expensive pipeline safely.

## Build Pages

```bash
python build_dataset.py prepare-pages \
  --bw-root raw/bw \
  --color-root raw/color \
  --output dataset \
  --text-detections work/magi_bw.json \
  --min-inlier-ratio 0.15
```

This step:

- removes text boxes when the surrounding background variance is low;
- aligns color pages onto BW pages with SIFT + brute-force matching + RANSAC;
- writes accepted page pairs into `dataset/pages/bw` and `dataset/pages/color`;
- records homography, inlier ratio, text boxes, and failures in `metadata.json`.

## Extract And Filter Panels

```bash
python build_dataset.py extract-panels \
  --dataset dataset \
  --panel-detections work/magi_pages.json \
  --min-ssim 0.55
```

With SAM refinement:

```bash
python build_dataset.py extract-panels \
  --dataset dataset \
  --panel-detections work/magi_pages.json \
  --sam-checkpoint sam_vit_h_4b8939.pth \
  --sam-model-type vit_h \
  --device cuda \
  --min-ssim 0.55
```

Panel filtering applies XDoG to BW/color panel crops and keeps pairs whose edge-map SSIM is high enough.

## Captions

The code can consume caption JSON files generated offline by Moondream, Florence-2, LLaVa, or any other VLM. Each JSON maps `panel_id` to text or `{ "caption": "..." }`.

```bash
python build_dataset.py caption-panels \
  --dataset dataset \
  --caption-sources work/moondream.json work/florence2.json work/llava.json
```

For each panel it writes:

```text
dataset/panels/captions/<panel_id>.json
```

with `plain` and `stable_diffusion_prompt` fields.

## Splits

```bash
python build_dataset.py build-splits \
  --dataset dataset \
  --test-in-ratio 0.1 \
  --test-out-ratio 0.15
```

`test_in` uses held-out chapters from titles present in training. `test_out` holds out complete titles to measure style generalization.

## One-Shot Run

```bash
python build_dataset.py run-all \
  --bw-root raw/bw \
  --color-root raw/color \
  --output dataset \
  --text-detections work/magi_bw.json \
  --panel-detections work/magi_pages.json \
  --caption-sources work/moondream.json work/florence2.json work/llava.json
```

## Training Hook

Use `dataset_pipeline.training_dataset.PairedMangaPanelDataset` to load `dataset/metadata.json` and panel pairs:

```python
from dataset_pipeline.training_dataset import PairedMangaPanelDataset

train = PairedMangaPanelDataset("dataset", split="train")
sample = train[0]
```

The sample contains `bw`, `color`, and `metadata`.

