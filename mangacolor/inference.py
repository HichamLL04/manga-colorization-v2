"""Shared inference for CLI, editor and pseudo-label generation."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from .checkpoints import load_model
from .reconstruction import (
    ColorField,
    Conditions,
    ExportConfig,
    ExportRecord,
    InferenceConfig,
    Masks,
    ReconstructionConfig,
    RenderResult,
    determine_export_dimensions,
    export_result,
    predict_color_field,
    reconstruct,
)


def open_rgb(path):
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(white, rgba).convert("RGB")


def parse_palette(colors):
    """Hex colors to RGB. Empty means no global conditioning."""
    result = []
    for color in colors or []:
        value = color.strip().lstrip("#")
        if len(value) != 6:
            raise ValueError(f"Color inválido: {color}. Usa #RRGGBB.")
        try:
            result.append(tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)))
        except ValueError as exc:
            raise ValueError(f"Color inválido: {color}") from exc
    if len(result) > 32:
        raise ValueError("La paleta admite hasta 32 colores.")
    return np.asarray(result, dtype=np.uint8).reshape(-1, 3)


def hints_tensor(rgba, size):
    """Resize premultiplied hints, preserving black and antialiased masks."""
    if rgba is None:
        return torch.zeros(4, size[1], size[0])
    data = np.asarray(rgba.convert("RGBA"), dtype=np.float32) / 255
    alpha = data[:, :, 3:4]
    premultiplied = (data[:, :, :3] * 2 - 1) * alpha
    values = np.concatenate([premultiplied, alpha], axis=2)
    values = torch.from_numpy(values.copy()).permute(2, 0, 1).unsqueeze(0)
    # Area interpolation retains the contribution of a one-pixel stroke when shrinking.
    if size[0] <= rgba.width and size[1] <= rgba.height:
        values = F.interpolate(values, size=(size[1], size[0]), mode="area")
    else:
        values = F.interpolate(values, size=(size[1], size[0]), mode="bilinear", align_corners=False)
    return values[0]


class ColorizationPipeline:
    def __init__(self, checkpoint, device="cpu"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA no está disponible en este intérprete de PyTorch.")
        self.device = torch.device(device)
        self.model, self.metadata = load_model(checkpoint, device=device)
        import hashlib
        digest = hashlib.sha256()
        with Path(checkpoint).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.metadata["checkpoint_sha256"] = digest.hexdigest()
        self.model.eval()

    @torch.inference_mode()
    def colorize(self, image, hints=None, palette=None, size=576,
                 palette_mode="hints", lock_hints=False, preserve_ink=False,
                 reconstruct_mode="legacy", export_size="original",
                 text_mask=None, ink_mask=None,
                 detail_strength=1.0, alpha_original=0.0):
        """palette_mode: learned (trained FiLM) or hints (two-pass original v2).
        reconstruct_mode: 'guided' (decoupled guided upsampling, SSD Section 11) or 'legacy'.
        """
        # If high-resolution guided reconstruction or custom masks/export size requested:
        if reconstruct_mode == "guided" or export_size != "original" or text_mask is not None or ink_mask is not None:
            conditions = Conditions(hints=hints, palette=palette, palette_mode=palette_mode)
            inf_cfg = InferenceConfig(canonical_size=size, device=str(self.device))
            cf = predict_color_field(image, conditions, self.model, inf_cfg, self.metadata)
            recon_cfg = ReconstructionConfig(
                export_size=export_size,
                detail_strength=detail_strength,
                alpha_original=alpha_original,
                preserve_ink=preserve_ink,
            )
            masks = Masks(
                text_mask=text_mask,
                ink_mask=ink_mask,
                hints_lock=hints,
                lock_hints=lock_hints,
            )
            render_res = reconstruct(image, cf, masks=masks, reconstruction_config=recon_cfg)
            report = {
                "palette_mode": conditions.palette_mode,
                "locked_regions": bool(lock_hints),
                "model_metadata": self.metadata,
                "inference_size": list(cf.canonical_size),
                "reconstruction_key": render_res.reconstruction_key,
                "color_field_key": cf.cache_key,
                "export_dimensions": list(render_res.dimensions),
                "stats": render_res.stats,
                "warnings": render_res.warnings,
            }
            return render_res.image, report

        if size < 32 or size > 2048:
            raise ValueError("size debe estar entre 32 y 2048 (lado mayor de inferencia).")
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size
        if hints is not None and hints.size != original_size:
            raise ValueError("Las pistas RGBA deben medir lo mismo que la imagen original.")
        ratio = size / max(original_size)
        width, height = [max(1, round(v * ratio)) for v in original_size]
        gray = image.convert("L").resize((width, height), Image.Resampling.LANCZOS)
        sketch = torch.from_numpy(np.asarray(gray, dtype=np.float32).copy() / 255)[None]
        hint = hints_tensor(hints, (width, height))
        pad = (0, (-width) % 32, 0, (-height) % 32)
        x = torch.cat([F.pad(sketch, pad, value=1), F.pad(hint, pad)], dim=0)[None].to(self.device)
        palette = np.asarray(palette if palette is not None else [], dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(palette).all() or (palette < 0).any() or (palette > 255).any():
            raise ValueError("La paleta debe contener RGB finitos entre 0 y 255.")
        if len(palette) > 32:
            raise ValueError("Máximo 32 colores por paleta.")
        if palette_mode not in ("hints", "learned"):
            raise ValueError("palette_mode debe ser hints o learned.")
        report = {"palette_mode": "none", "locked_regions": bool(lock_hints),
                  "model_metadata": self.metadata, "inference_size": [width, height]}
        if len(palette) and palette_mode == "learned":
            if not self.metadata.get("palette_trained", False):
                raise ValueError("Este checkpoint es v2 nativo. Usa palette-mode=hints o entrena el adaptador con mangacolor train.")
            pal = torch.from_numpy(palette / 127.5 - 1)[None].to(self.device)
            fake, _ = self.model(x, pal, torch.ones(1, len(palette), device=self.device))
            report["palette_mode"] = "learned"
        else:
            fake, _ = self.model(x)
            if len(palette):
                # Bootstrap a global color preference through the native local-hint input.
                # Sparse samples avoid replacing every pixel or claiming semantic labels.
                colors = torch.from_numpy(palette / 127.5 - 1).to(self.device)
                step = max(4, min(width, height) // 24)
                ys = torch.arange(step // 2, height, step, device=self.device)
                xs = torch.arange(step // 2, width, step, device=self.device)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                samples = fake[0, :, yy, xx].permute(1, 2, 0)
                # Compare chromatic components so a dark shirt can receive a bright hue.
                chroma = samples - samples.mean(-1, keepdim=True)
                pal_chroma = colors - colors.mean(-1, keepdim=True)
                nearest = ((chroma[:, :, None] - pal_chroma) ** 2).sum(-1).argmin(-1)
                valid = ((x[0, 0, yy, xx] > .12)
                         & (samples.max(-1).values - samples.min(-1).values > .08)
                         & (x[0, 4, yy, xx] == 0))
                yv, xv = yy[valid], xx[valid]
                x[0, 1:4, yv, xv] = colors[nearest[valid]].T
                x[0, 4, yv, xv] = 1
                fake, _ = self.model(x)
                report["palette_mode"] = "native_sparse_hints"
                report["palette_samples"] = int(valid.sum())
        output = fake[0, :, :height, :width].clamp(-1, 1).add(1).mul(127.5)
        array = output.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
        result = Image.fromarray(array).resize(original_size, Image.Resampling.LANCZOS)
        if preserve_ink:
            array = np.asarray(result, dtype=np.float32)
            ink = np.asarray(image.convert("L"), dtype=np.float32) / 255
            factor = np.clip(ink / .25, 0, 1)
            result = Image.fromarray((array * factor[:, :, None]).round().astype(np.uint8))
        if lock_hints and hints is not None:
            # Last: opaque locked pixels take precedence over ink preservation.
            result = Image.composite(hints.convert("RGB"), result, hints.getchannel("A"))
        return result, report

    def predict_color_field(self, image, hints=None, palette=None, size=576, palette_mode="hints"):
        """Calculate canonical ColorField without rendering to export dimensions (SSD 17.2)."""
        conditions = Conditions(hints=hints, palette=palette, palette_mode=palette_mode)
        inf_cfg = InferenceConfig(canonical_size=size, device=str(self.device))
        return predict_color_field(image, conditions, self.model, inf_cfg, self.metadata)

    def reconstruct(self, image, color_field, masks=None, reconstruction_config=None):
        """Render high-resolution output from ColorField (SSD 17.2)."""
        return reconstruct(image, color_field, masks=masks, reconstruction_config=reconstruction_config)

    def export_result(self, render_result, destination, export_config=None):
        """Export image and report without overwriting (OUT01 / SSD 17.2)."""
        return export_result(render_result, destination, export_config=export_config)



def pseudo_label(manifest, output, pipeline, size=576):
    """Never relabel pseudo-targets as human ground truth; keep unlabeled val out."""
    import json
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.jsonl"
    if destination.exists():
        raise FileExistsError(f"Ya existe {destination}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    reserved = {row.get(key) for row in rows if row.get("split") == "val"
                for key in ("source_group", "split_group", "group") if row.get(key)}
    checkpoint_meta = pipeline.metadata
    for row in rows:
        source = (manifest.parent / row["input"]).resolve()
        row["input"] = str(source)
        if row.get("target"):
            row["target"] = str((manifest.parent / row["target"]).resolve())
        elif (row.get("split") != "val" and not any(row.get(key) in reserved
              for key in ("source_group", "split_group", "group") if row.get(key))):
            result, _ = pipeline.colorize(open_rgb(source), size=size)
            name = f"{row['id']}_{row.get('variant', 'bw')}_pseudo.png"
            target = output / name
            if target.exists():
                raise FileExistsError(target)
            result.save(target)
            row.update(target=name, split="train", supervision="pseudo", weight=.25,
                       teacher=checkpoint_meta)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return destination
