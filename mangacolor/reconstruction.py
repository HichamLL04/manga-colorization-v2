"""Decoupled high-resolution reconstruction and canonical color fields.

Implements MangaColor SSD Milestone B (RES01 to RES06, Section 11 and Section 17.2):
- predict_color_field: calculates and caches canonical color field at canonical size.
- reconstruct: guided chrominance upsampling, luminance decomposition, gamut mapping,
  and composition priority order (reconstruction -> exact hint locks -> protected masks).
- export_result: non-overwriting file export with JSON audit report.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from .color import (
    cielab_to_srgb,
    map_gamut,
    srgb_to_cielab,
    validate_finite,
)
from .guided_filter import (
    guided_filter_tiled_torch,
    smooth_filter_np,
)


@dataclass
class Conditions:
    """Inputs that condition the neural network prediction."""
    hints: Optional[Image.Image] = None
    palette: Optional[Union[List[str], np.ndarray]] = None
    palette_mode: str = "hints"  # "hints" or "learned"


@dataclass
class InferenceConfig:
    """Parameters for canonical color field prediction."""
    canonical_size: int = 576
    preprocess_profile: str = "canonical_long_edge_v1"
    device: str = "cpu"
    seed: int = 42


@dataclass
class ColorField:
    """Canonical color prediction tensor and provenance metadata (SSD 11.7)."""
    rgb_canonical: np.ndarray  # float32 in [0, 1], shape (H, W, 3)
    lab_canonical: np.ndarray  # float32, L* in [0, 100], shape (H, W, 3)
    canonical_size: Tuple[int, int]  # (width, height)
    pad: Tuple[int, int, int, int]  # (left, right, top, bottom)
    original_size: Tuple[int, int]  # (width, height)
    cache_key: str
    original_pixel_sha256: str
    checkpoint_sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Persist canonical field to NPZ without pickle (SSD 11.7)."""
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps({
            "canonical_size": list(self.canonical_size),
            "pad": list(self.pad),
            "original_size": list(self.original_size),
            "cache_key": self.cache_key,
            "original_pixel_sha256": self.original_pixel_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metadata": self.metadata,
        }, ensure_ascii=False).encode("utf-8")
        meta_array = np.frombuffer(meta_json, dtype=np.uint8)

        # allow_pickle=False guarantees safe and portable serialization
        np.savez(
            str(destination),
            rgb=self.rgb_canonical.astype(np.float32),
            lab=self.lab_canonical.astype(np.float32),
            metadata=meta_array,
        )
        return destination

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "ColorField":
        """Load canonical field from NPZ with allow_pickle=False."""
        source = Path(path).resolve()
        with np.load(str(source), allow_pickle=False) as data:
            rgb = data["rgb"].astype(np.float32)
            lab = data["lab"].astype(np.float32)
            meta_raw = data["metadata"].tobytes().decode("utf-8")
            meta = json.loads(meta_raw)

        return cls(
            rgb_canonical=rgb,
            lab_canonical=lab,
            canonical_size=tuple(meta["canonical_size"]),
            pad=tuple(meta["pad"]),
            original_size=tuple(meta["original_size"]),
            cache_key=meta["cache_key"],
            original_pixel_sha256=meta["original_pixel_sha256"],
            checkpoint_sha256=meta["checkpoint_sha256"],
            metadata=meta.get("metadata", {}),
        )


@dataclass
class Masks:
    """Masks for protection and locks during reconstruction (SSD 11.6)."""
    text_mask: Optional[Image.Image] = None
    ink_mask: Optional[Image.Image] = None
    hints_lock: Optional[Image.Image] = None
    lock_hints: bool = False


@dataclass
class ReconstructionConfig:
    """Parameters for guided high-resolution reconstruction (SSD Section 11)."""
    export_size: Union[str, int, Tuple[int, int]] = "original"
    guided_filter_radius: int = 4
    guided_filter_eps: float = 1e-3
    tile_size: int = 1024
    detail_strength: float = 1.0  # SSD 11.4: default 1.0
    alpha_original: float = 0.0  # SSD 11.4: default 0.0 (1.0 = original luminance)
    smoothing_filter: str = "gaussian_v1"  # "box_v1" or "gaussian_v1"
    smoothing_radius: int = 5
    preserve_ink: bool = False
    gamut_mapping: bool = True
    gamut_max_iterations: int = 8
    reconstruction_version: str = "reconstruct_v1"


@dataclass
class RenderResult:
    """Result of high-resolution reconstruction."""
    image: Image.Image
    dimensions: Tuple[int, int]
    warnings: List[str]
    stats: Dict[str, Any]
    reconstruction_key: str
    reconstruction_config: Dict[str, Any]
    color_field_key: str


@dataclass
class ExportConfig:
    """Export configuration."""
    overwrite: bool = False
    save_report: bool = True
    format: str = "PNG"


@dataclass
class ExportRecord:
    """Record of an exported image and companion report (SSD 17.2)."""
    path: Path
    sha256: str
    dimensions: Tuple[int, int]
    report_path: Optional[Path]
    report: Dict[str, Any]


def compute_sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def compute_image_pixels_sha256(image: Image.Image) -> str:
    """Hash deterministic raw pixels of normalized RGB image."""
    rgb = image.convert("RGB")
    return compute_sha256(rgb.tobytes())


def compute_canonical_prediction_key(
    original_sha256: str,
    checkpoint_sha256: str,
    canonical_size: int,
    palette: Optional[np.ndarray],
    hints_hash: Optional[str],
    preprocess_profile: str,
    palette_mode: str,
) -> str:
    key_material = {
        "original_sha256": original_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "canonical_size": canonical_size,
        "palette": palette.tolist() if palette is not None else [],
        "hints_hash": hints_hash,
        "preprocess_profile": preprocess_profile,
        "palette_mode": palette_mode,
    }
    encoded = json.dumps(key_material, sort_keys=True).encode("utf-8")
    return compute_sha256(encoded)


def compute_reconstruction_key(
    color_field_key: str,
    dimensions: Tuple[int, int],
    reconstruction_config: Dict[str, Any],
    masks_hash: str,
) -> str:
    key_material = {
        "color_field_key": color_field_key,
        "dimensions": list(dimensions),
        "reconstruction_config": reconstruction_config,
        "masks_hash": masks_hash,
    }
    encoded = json.dumps(key_material, sort_keys=True).encode("utf-8")
    return compute_sha256(encoded)


def predict_color_field(
    original: Image.Image,
    conditions: Conditions,
    model: Any,
    inference_config: Optional[InferenceConfig] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> ColorField:
    """Predict canonical color field from original image and conditions (SSD 17.2).

    Runs inference at canonical size (profile canonical_long_edge_v1).
    Does NOT depend on export dimensions (RES01, RES02).
    """
    if inference_config is None:
        inference_config = InferenceConfig()

    image_clean = ImageOps.exif_transpose(original).convert("RGB")
    orig_w, orig_h = image_clean.size
    orig_pixel_hash = compute_image_pixels_sha256(image_clean)

    canonical_size = inference_config.canonical_size
    if canonical_size < 32 or canonical_size > 2048:
        raise ValueError("canonical_size debe estar entre 32 y 2048.")

    ratio = canonical_size / max(orig_w, orig_h)
    width = max(1, round(orig_w * ratio))
    height = max(1, round(orig_h * ratio))

    # Preprocessing: Lanczos grayscale and zero-padded hints
    gray = image_clean.convert("L").resize((width, height), Image.Resampling.LANCZOS)
    sketch = torch.from_numpy(np.asarray(gray, dtype=np.float32).copy() / 255.0)[None]

    hints_img = conditions.hints
    hints_hash = None
    if hints_img is not None:
        if hints_img.size != (orig_w, orig_h):
            raise ValueError("Las pistas RGBA deben medir lo mismo que la imagen original.")
        hints_hash = compute_sha256(hints_img.convert("RGBA").tobytes())

    # Build hint tensor
    from .inference import hints_tensor
    hint_t = hints_tensor(hints_img, (width, height))

    pad = (0, (-width) % 32, 0, (-height) % 32)
    device = torch.device(inference_config.device)
    x = torch.cat([F.pad(sketch, pad, value=1), F.pad(hint_t, pad)], dim=0)[None].to(device)

    palette_arr = np.asarray(conditions.palette if conditions.palette is not None else [], dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(palette_arr).all() or (palette_arr < 0).any() or (palette_arr > 255).any():
        raise ValueError("La paleta debe contener RGB finitos entre 0 y 255.")
    if len(palette_arr) > 32:
        raise ValueError("Máximo 32 colores por paleta.")

    meta = checkpoint_metadata or getattr(model, "metadata", {}) or {}
    ckpt_hash = meta.get("checkpoint_sha256", "unknown_checkpoint")

    # Run neural prediction
    with torch.inference_mode():
        if len(palette_arr) and conditions.palette_mode == "learned":
            if not meta.get("palette_trained", False):
                raise ValueError("Este checkpoint es v2 nativo. Usa palette-mode=hints o entrena el adaptador.")
            pal = torch.from_numpy(palette_arr / 127.5 - 1.0)[None].to(device)
            fake, _ = model(x, pal, torch.ones(1, len(palette_arr), device=device))
        else:
            fake, _ = model(x)
            if len(palette_arr):
                # Native sparse hints bootstrap (SSD Section 10.4)
                colors = torch.from_numpy(palette_arr / 127.5 - 1.0).to(device)
                step = max(4, min(width, height) // 24)
                ys = torch.arange(step // 2, height, step, device=device)
                xs = torch.arange(step // 2, width, step, device=device)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                samples = fake[0, :, yy, xx].permute(1, 2, 0)
                chroma = samples - samples.mean(-1, keepdim=True)
                pal_chroma = colors - colors.mean(-1, keepdim=True)
                nearest = ((chroma[:, :, None] - pal_chroma) ** 2).sum(-1).argmin(-1)
                valid = ((x[0, 0, yy, xx] > 0.12)
                         & (samples.max(-1).values - samples.min(-1).values > 0.08)
                         & (x[0, 4, yy, xx] == 0))
                yv, xv = yy[valid], xx[valid]
                x[0, 1:4, yv, xv] = colors[nearest[valid]].T
                x[0, 4, yv, xv] = 1
                fake, _ = model(x)

    # Slice out padding and convert to float32 [0, 1]
    output = fake[0, :, :height, :width].clamp(-1.0, 1.0).add(1.0).mul(0.5)
    rgb_canonical = output.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    lab_canonical = srgb_to_cielab(rgb_canonical)

    cache_key = compute_canonical_prediction_key(
        original_sha256=orig_pixel_hash,
        checkpoint_sha256=ckpt_hash,
        canonical_size=canonical_size,
        palette=palette_arr if len(palette_arr) else None,
        hints_hash=hints_hash,
        preprocess_profile=inference_config.preprocess_profile,
        palette_mode=conditions.palette_mode,
    )

    provenance = {
        "palette_mode": conditions.palette_mode,
        "preprocess_profile": inference_config.preprocess_profile,
        "canonical_size": [width, height],
        "device": str(device),
        "model_metadata": meta,
    }

    return ColorField(
        rgb_canonical=rgb_canonical,
        lab_canonical=lab_canonical,
        canonical_size=(width, height),
        pad=pad,
        original_size=(orig_w, orig_h),
        cache_key=cache_key,
        original_pixel_sha256=orig_pixel_hash,
        checkpoint_sha256=ckpt_hash,
        metadata=provenance,
    )


def determine_export_dimensions(
    original_size: Tuple[int, int],
    export_size: Union[str, int, Tuple[int, int]],
) -> Tuple[int, int]:
    orig_w, orig_h = original_size
    if export_size == "original":
        return orig_w, orig_h
    elif isinstance(export_size, int):
        if export_size < 1:
            raise ValueError("export_size entero debe ser positivo.")
        ratio = export_size / max(orig_w, orig_h)
        return max(1, round(orig_w * ratio)), max(1, round(orig_h * ratio))
    elif isinstance(export_size, (tuple, list)) and len(export_size) == 2:
        w, h = int(export_size[0]), int(export_size[1])
        if w < 1 or h < 1:
            raise ValueError("Las dimensiones de exportación deben ser positivas.")
        return w, h
    else:
        raise ValueError(f"Formato de export_size inválido: {export_size}")


def reconstruct(
    original: Image.Image,
    color_field: ColorField,
    masks: Optional[Masks] = None,
    reconstruction_config: Optional[ReconstructionConfig] = None,
) -> RenderResult:
    """Reconstruct high-resolution color using Guided Filtering and luminance detail (SSD 11).

    Enforces composition priority order (SSD 11.6):
    1. Base reconstruction (Guided a*, b* and luminance detail preservation)
    2. Locked hints exact RGB composition
    3. Protected ink and text masks (exact pixel restoration from original).
    """
    if reconstruction_config is None:
        reconstruction_config = ReconstructionConfig()
    if masks is None:
        masks = Masks()

    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    image_clean = ImageOps.exif_transpose(original).convert("RGB")
    target_w, target_h = determine_export_dimensions(
        image_clean.size,
        reconstruction_config.export_size,
    )

    # Scale original image to target resolution if needed
    if (target_w, target_h) == image_clean.size:
        target_orig_img = image_clean
    else:
        target_orig_img = image_clean.resize((target_w, target_h), Image.Resampling.LANCZOS)

    target_orig_rgb = np.asarray(target_orig_img, dtype=np.float32) / 255.0
    target_orig_lab = srgb_to_cielab(target_orig_rgb)
    L_original = target_orig_lab[:, :, 0]  # [0, 100]

    # 1. Upsample canonical chrominance a*, b* to target resolution
    ab_canon = color_field.lab_canonical[:, :, 1:3]  # (H_c, W_c, 2)
    ab_tensor = torch.from_numpy(ab_canon).permute(2, 0, 1).unsqueeze(0).float()
    ab_upsampled = F.interpolate(
        ab_tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    # 2. Guided Filter on a*, b* using original grayscale as structural guide
    guide_I = target_orig_rgb.mean(axis=-1, keepdims=True)  # (H, W, 1) in [0, 1]
    guide_tensor = torch.from_numpy(guide_I).permute(2, 0, 1).unsqueeze(0).float()

    ab_guided = guided_filter_tiled_torch(
        guide=guide_tensor,
        src=ab_upsampled,
        r=reconstruction_config.guided_filter_radius,
        eps=reconstruction_config.guided_filter_eps,
        tile_size=reconstruction_config.tile_size,
    )
    ab_guided_np = ab_guided.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

    # 3. Luminance and Detail Decomposition (SSD Section 11.4)
    # L_original = luminosidad del original en resolución completa
    # B_original = filtro_suave_versionado(L_original)
    # D_original = L_original - B_original
    # B_predicho = ampliar_suavemente(L_predicho_canonico)
    # L_mezclado = clip(B_predicho + detalle_strength * D_original, 0, 100)
    # L_final = alpha_original * L_original + (1 - alpha_original) * L_mezclado
    B_original = smooth_filter_np(
        L_original,
        radius=reconstruction_config.smoothing_radius,
        filter_type=reconstruction_config.smoothing_filter,
    )
    D_original = L_original - B_original

    L_pred_canon = color_field.lab_canonical[:, :, 0:1]  # (H_c, W_c, 1)
    L_pred_tensor = torch.from_numpy(L_pred_canon).permute(2, 0, 1).unsqueeze(0).float()
    B_pred_tensor = F.interpolate(
        L_pred_tensor,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    )
    B_predicho = B_pred_tensor.squeeze().cpu().numpy()

    detalle_k = reconstruction_config.detail_strength
    alpha_orig = reconstruction_config.alpha_original

    L_mezclado = np.clip(B_predicho + detalle_k * D_original, 0.0, 100.0)
    L_final = alpha_orig * L_original + (1.0 - alpha_orig) * L_mezclado

    # 4. Recompose CIELAB and perform Gamut Mapping (SSD Section 11.5)
    lab_reconstructed = np.concatenate([L_final[:, :, None], ab_guided_np], axis=-1)

    if reconstruction_config.gamut_mapping:
        rgb_reconstructed, lab_mapped, affected_fraction = map_gamut(
            lab_reconstructed,
            iterations=reconstruction_config.gamut_max_iterations,
        )
        stats["gamut_affected_fraction"] = affected_fraction
        if affected_fraction > 0.05:
            warnings.append(
                f"Gamut mapping afectó al {affected_fraction * 100:.2f}% de los píxeles."
            )
    else:
        rgb_reconstructed = np.clip(cielab_to_srgb(lab_reconstructed), 0.0, 1.0)
        stats["gamut_affected_fraction"] = 0.0

    # 5. Optional heuristic ink preservation
    if reconstruction_config.preserve_ink:
        ink = target_orig_rgb.mean(axis=-1)
        factor = np.clip(ink / 0.25, 0.0, 1.0)
        rgb_reconstructed = rgb_reconstructed * factor[:, :, None]

    # 6. Priority Composition (SSD Section 11.6)
    # Priority: Reconstructed -> Exact Locked Hints -> Protected Ink / Text
    locked_pixels_count = 0
    if masks.lock_hints and masks.hints_lock is not None:
        hints_clean = masks.hints_lock.convert("RGBA")
        if hints_clean.size != (target_w, target_h):
            hints_clean = hints_clean.resize((target_w, target_h), Image.Resampling.BILINEAR)
        hints_np = np.asarray(hints_clean, dtype=np.float32) / 255.0
        alpha = hints_np[:, :, 3:4]
        hint_rgb = hints_np[:, :, :3]
        # Composite exact lock
        rgb_reconstructed = (1.0 - alpha) * rgb_reconstructed + alpha * hint_rgb
        locked_pixels_count = int(np.count_nonzero(alpha > 0.5))
    stats["locked_pixels"] = locked_pixels_count

    # Priority 3 & 4: Protected Masks (ALWAYS PREVAIL - SSD Section 11.6 & AT08/AT09)
    ink_protected_count = 0
    if masks.ink_mask is not None:
        ink_m = masks.ink_mask.convert("L")
        if ink_m.size != (target_w, target_h):
            ink_m = ink_m.resize((target_w, target_h), Image.Resampling.NEAREST)
        ink_arr = np.asarray(ink_m) > 128
        rgb_reconstructed[ink_arr] = target_orig_rgb[ink_arr]
        ink_protected_count = int(np.count_nonzero(ink_arr))
    stats["ink_pixels_protected"] = ink_protected_count

    text_protected_count = 0
    if masks.text_mask is not None:
        text_m = masks.text_mask.convert("L")
        if text_m.size != (target_w, target_h):
            text_m = text_m.resize((target_w, target_h), Image.Resampling.NEAREST)
        text_arr = np.asarray(text_m) > 128
        # Exact restoration from original pixels guarantees AT08: error RGB máximo cero
        rgb_reconstructed[text_arr] = target_orig_rgb[text_arr]
        text_protected_count = int(np.count_nonzero(text_arr))
    stats["text_pixels_protected"] = text_protected_count

    # Validate output numerical integrity
    validate_finite(rgb_reconstructed, "rgb_reconstructed")
    final_uint8 = np.clip(np.round(rgb_reconstructed * 255.0), 0, 255).astype(np.uint8)
    rendered_image = Image.fromarray(final_uint8, mode="RGB")

    # Hashes and reconstruction key
    masks_hash_material = {
        "text_mask": compute_sha256(masks.text_mask.tobytes()) if masks.text_mask else None,
        "ink_mask": compute_sha256(masks.ink_mask.tobytes()) if masks.ink_mask else None,
        "hints_lock": compute_sha256(masks.hints_lock.tobytes()) if (masks.lock_hints and masks.hints_lock) else None,
    }
    masks_hash = compute_sha256(json.dumps(masks_hash_material, sort_keys=True).encode("utf-8"))

    recon_config_dict = asdict(reconstruction_config)
    recon_key = compute_reconstruction_key(
        color_field_key=color_field.cache_key,
        dimensions=(target_w, target_h),
        reconstruction_config=recon_config_dict,
        masks_hash=masks_hash,
    )

    return RenderResult(
        image=rendered_image,
        dimensions=(target_w, target_h),
        warnings=warnings,
        stats=stats,
        reconstruction_key=recon_key,
        reconstruction_config=recon_config_dict,
        color_field_key=color_field.cache_key,
    )


def export_result(
    render_result: RenderResult,
    destination: Union[str, Path],
    export_config: Optional[ExportConfig] = None,
) -> ExportRecord:
    """Export reconstructed image and companion JSON audit report (SSD 17.2 & OUT01).

    Fails with FileExistsError if destination already exists and overwrite=False.
    """
    if export_config is None:
        export_config = ExportConfig()

    dest_path = Path(destination).resolve()
    if dest_path.exists() and not export_config.overwrite:
        raise FileExistsError(f"Ya existe {dest_path}; OUT01 prohíbe sobrescribir sin permiso.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    render_result.image.save(dest_path, format=export_config.format)

    # Compute digest of exported file
    file_sha256 = compute_sha256(dest_path.read_bytes())

    report_path = None
    report_dict: Dict[str, Any] = {
        "destination": str(dest_path),
        "sha256": file_sha256,
        "dimensions": list(render_result.dimensions),
        "color_field_key": render_result.color_field_key,
        "reconstruction_key": render_result.reconstruction_key,
        "stats": render_result.stats,
        "warnings": render_result.warnings,
        "reconstruction_config": render_result.reconstruction_config,
    }

    if export_config.save_report:
        report_path = dest_path.with_suffix(dest_path.suffix + ".json")
        report_path.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return ExportRecord(
        path=dest_path,
        sha256=file_sha256,
        dimensions=render_result.dimensions,
        report_path=report_path,
        report=report_dict,
    )
