# Training A V2-Compatible Model

Use this when you have built a dataset with `build_dataset.py` and want weights that can be loaded by the existing V2 inference code.

The training entrypoint is:

```bash
python train_v2.py \
  --dataset dataset \
  --output-dir runs/v2_aligned \
  --gpu \
  --batch-size 4 \
  --epochs 15 \
  --crop-size 512 \
  --input-channels 5
```

The produced file `runs/v2_aligned/generator_latest.pt` is a `networks.models.Generator` state dict. Use it in inference like this:

```bash
python inference.py \
  -p path/to/page.png \
  -gen runs/v2_aligned/generator_latest.pt \
  --autohint \
  -g
```

## 5 Or 6 Input Channels

For maximum compatibility with the published V2 code path, train with:

```bash
--input-channels 5
```

This uses:

```text
BW image (1) + color hint RGB/mask (4) = 5 channels
```

To train the paper-style DFM variant:

```bash
--input-channels 6
```

This uses:

```text
BW image (1) + DFM (1) + color hint RGB/mask (4) = 6 channels
```

`MangaColorizator` detects this from `to0.0.weight` in the checkpoint. A 6-channel checkpoint automatically enables DFM generation at inference time.

## Extractor Weights

If you have compatible extractor weights:

```bash
python train_v2.py \
  --dataset dataset \
  --output-dir runs/v2_aligned \
  --extractor-weights networks/extractor.pth \
  --gpu
```

When extractor weights are passed, the encoder is frozen by default. Without extractor weights, the encoder is trained from scratch, because freezing a random encoder would make training mostly useless.

## Optional Perceptual Loss

The V1 training code used VGG perceptual loss. In this V2-compatible trainer it is off by default so training does not silently use random VGG weights.

Enable it only with real VGG-16 weights:

```bash
python train_v2.py \
  --dataset dataset \
  --vgg-weights path/to/vgg16-397923af.pth \
  --content-weight 1.0
```

## Losses

The trainer keeps the useful V1 ingredients while using the V2 generator:

- alternating discriminator/generator updates;
- L1 loss on main output;
- auxiliary output L1 loss;
- adversarial loss;
- optional VGG perceptual loss;
- white color penalty;
- hint sampling with empty, sparse, and full masks.

