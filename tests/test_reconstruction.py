import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from mangacolor.color import (
    cielab_to_srgb,
    map_gamut,
    srgb_to_cielab,
    validate_finite,
)
from mangacolor.guided_filter import (
    guided_filter_np,
    guided_filter_tiled_torch,
    smooth_filter_np,
)
from mangacolor.reconstruction import (
    ColorField,
    Conditions,
    ExportConfig,
    InferenceConfig,
    Masks,
    ReconstructionConfig,
    export_result,
    reconstruct,
)


def test_color_conversions_accuracy_and_reversibility():
    """Verify sRGB <-> XYZ <-> CIELAB numerical accuracy and reference patches."""
    patches = np.array([
        [1.0, 1.0, 1.0],  # White
        [0.0, 0.0, 0.0],  # Black
        [0.5, 0.5, 0.5],  # Middle gray
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.0, 1.0],  # Magenta
    ], dtype=np.float32)

    lab = srgb_to_cielab(patches)

    # Check reference D65 white point: L* = 100, a* ~ 0, b* ~ 0
    assert abs(lab[0, 0] - 100.0) < 1e-4
    assert abs(lab[0, 1]) < 1e-3
    assert abs(lab[0, 2]) < 1e-3

    # Check black: L* = 0, a* = 0, b* = 0
    assert abs(lab[1, 0]) < 1e-5
    assert abs(lab[1, 1]) < 1e-5
    assert abs(lab[1, 2]) < 1e-5

    # Roundtrip accuracy
    reconstructed = cielab_to_srgb(lab)
    np.testing.assert_allclose(reconstructed, patches, atol=1e-5)


def test_gamut_mapping_preserves_lightness_and_maps_into_srgb():
    """Verify gamut mapping reduces chroma towards neutral axis at constant L*."""
    # Out-of-gamut Lab point: bright and extremely saturated
    out_of_gamut_lab = np.array([[[90.0, 110.0, 95.0]]], dtype=np.float32)
    srgb, mapped_lab, frac = map_gamut(out_of_gamut_lab, iterations=12)

    assert frac == 1.0
    # Must be inside [0, 1]
    assert np.all(srgb >= 0.0) and np.all(srgb <= 1.0)
    # Lightness L* must remain constant
    assert abs(mapped_lab[0, 0, 0] - 90.0) < 1e-5

    # In-gamut point must be untouched
    in_gamut_lab = srgb_to_cielab(np.array([[[0.4, 0.5, 0.6]]], dtype=np.float32))
    srgb_in, mapped_lab_in, frac_in = map_gamut(in_gamut_lab)
    assert frac_in == 0.0
    np.testing.assert_allclose(srgb_in, [[[0.4, 0.5, 0.6]]], atol=1e-4)

    # Mixed array: in-gamut pixel must NOT lose its color when out-of-gamut pixel is present
    mixed_lab = np.concatenate([in_gamut_lab, out_of_gamut_lab], axis=1)
    srgb_mixed, _, frac_mixed = map_gamut(mixed_lab)
    assert frac_mixed > 0
    np.testing.assert_allclose(srgb_mixed[0, 0], [0.4, 0.5, 0.6], atol=1e-3)


def test_guided_filter_tiled_at10_seamlessness():
    """AT10: Seamless tiled Guided Filter with margin >= 2*r produces zero tile seams."""
    r = 4
    torch.manual_seed(42)
    guide = torch.rand(1, 1, 120, 120)
    src = torch.rand(1, 2, 120, 120)

    # Full filtering
    from mangacolor.guided_filter import guided_filter_torch
    full = guided_filter_torch(guide, src, r=r, eps=1e-3)

    # Tiled filtering with tile_size=40 and margin=16 (>= 2*r)
    tiled = guided_filter_tiled_torch(guide, src, r=r, eps=1e-3, tile_size=40, margin=16)

    max_diff = float((full - tiled).abs().max())
    # AT10 requirement: Sin costuras sintéticas mayores de 1 nivel RGB (1/255 ~= 0.0039)
    assert max_diff < 1e-5


def test_color_field_serialization_npz_allow_pickle_false(tmp_path):
    """RES06 & Section 11.7: NPZ serialization without pickle."""
    rgb = np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)
    lab = srgb_to_cielab(rgb)
    cf = ColorField(
        rgb_canonical=rgb,
        lab_canonical=lab,
        canonical_size=(64, 64),
        pad=(0, 0, 0, 0),
        original_size=(128, 128),
        cache_key="mock_cache_key_123",
        original_pixel_sha256="mock_orig_sha256",
        checkpoint_sha256="mock_ckpt_sha256",
        metadata={"note": "serialization_test", "version": 1},
    )

    save_path = tmp_path / "field.npz"
    cf.save_npz(save_path)
    assert save_path.exists()

    loaded = ColorField.load_npz(save_path)
    np.testing.assert_array_equal(loaded.rgb_canonical, cf.rgb_canonical)
    np.testing.assert_array_equal(loaded.lab_canonical, cf.lab_canonical)
    assert loaded.canonical_size == cf.canonical_size
    assert loaded.pad == cf.pad
    assert loaded.original_size == cf.original_size
    assert loaded.cache_key == cf.cache_key
    assert loaded.metadata == cf.metadata


def test_at07_export_multiple_sizes_reuses_same_color_field():
    """AT07 / RES01, RES02: Export to different sizes with zero network calls and identical cache key."""
    # Synthetic original image (100 x 80)
    orig = Image.new("RGB", (100, 80), (230, 230, 230))
    draw = ImageDraw.Draw(orig)
    draw.line((10, 10, 90, 70), fill=(20, 20, 20), width=3)

    # Canonical color field at 50 x 40
    rgb_c = np.ones((40, 50, 3), dtype=np.float32) * 0.7
    lab_c = srgb_to_cielab(rgb_c)
    cf = ColorField(
        rgb_canonical=rgb_c,
        lab_canonical=lab_c,
        canonical_size=(50, 40),
        pad=(0, 0, 0, 0),
        original_size=orig.size,
        cache_key="sha256_canonical_field_at07",
        original_pixel_sha256="sha256_orig",
        checkpoint_sha256="sha256_ckpt",
    )

    # Reconstruct at 3 sizes: original (100, 80), half (50, 40), double (200, 160)
    res_orig = reconstruct(orig, cf, reconstruction_config=ReconstructionConfig(export_size="original"))
    res_half = reconstruct(orig, cf, reconstruction_config=ReconstructionConfig(export_size=50))
    res_double = reconstruct(orig, cf, reconstruction_config=ReconstructionConfig(export_size=200))

    # Verification: identical color_field_key, matching dimensions
    assert res_orig.color_field_key == cf.cache_key
    assert res_half.color_field_key == cf.cache_key
    assert res_double.color_field_key == cf.cache_key

    assert res_orig.dimensions == (100, 80)
    assert res_half.dimensions == (50, 40)
    assert res_double.dimensions == (200, 160)


def test_at08_protected_text_exact_zero_error():
    """AT08 / RES04: Text inside binary mask has exactly 0.0 RGB error relative to original."""
    orig = Image.new("RGB", (80, 80), (250, 250, 250))
    draw = ImageDraw.Draw(orig)
    # Draw dark text/bubble
    draw.rectangle((20, 20, 50, 40), fill=(15, 15, 15))

    # Text protection mask covers the text rectangle
    text_mask = Image.new("L", orig.size, 0)
    text_draw = ImageDraw.Draw(text_mask)
    text_draw.rectangle((20, 20, 50, 40), fill=255)

    # Prediction wants to color everything bright red
    rgb_c = np.zeros((40, 40, 3), dtype=np.float32)
    rgb_c[:, :, 0] = 1.0  # Red
    lab_c = srgb_to_cielab(rgb_c)
    cf = ColorField(
        rgb_canonical=rgb_c,
        lab_canonical=lab_c,
        canonical_size=(40, 40),
        pad=(0, 0, 0, 0),
        original_size=orig.size,
        cache_key="cf_text_test",
        original_pixel_sha256="orig_hash",
        checkpoint_sha256="ckpt_hash",
    )

    # Also provide conflicting locked hint to test priority order (text mask must prevail!)
    hints = Image.new("RGBA", orig.size, (0, 0, 255, 255))  # Blue hints lock

    masks = Masks(text_mask=text_mask, hints_lock=hints, lock_hints=True)
    res = reconstruct(orig, cf, masks=masks, reconstruction_config=ReconstructionConfig(export_size="original"))

    orig_arr = np.asarray(orig)
    res_arr = np.asarray(res.image)

    # Within the text mask, error must be STRICTLY zero
    text_pixels = np.asarray(text_mask) > 128
    diff = np.abs(orig_arr[text_pixels].astype(int) - res_arr[text_pixels].astype(int))
    max_error = np.max(diff)
    assert max_error == 0, f"AT08 failed: max error {max_error} > 0"
    assert res.stats["text_pixels_protected"] == int(np.count_nonzero(text_pixels))


def test_at09_protected_ink_exact_zero_error():
    """AT09 / RES03: Ink inside explicit ink mask has exactly 0.0 RGB error relative to original."""
    orig = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(orig)
    draw.line((5, 5, 55, 55), fill=(10, 10, 10), width=4)

    ink_mask = Image.new("L", orig.size, 0)
    ink_draw = ImageDraw.Draw(ink_mask)
    ink_draw.line((5, 5, 55, 55), fill=255, width=4)

    rgb_c = np.zeros((32, 32, 3), dtype=np.float32)
    rgb_c[:, :, 1] = 0.8  # Green prediction
    lab_c = srgb_to_cielab(rgb_c)
    cf = ColorField(
        rgb_canonical=rgb_c,
        lab_canonical=lab_c,
        canonical_size=(32, 32),
        pad=(0, 0, 0, 0),
        original_size=orig.size,
        cache_key="cf_ink_test",
        original_pixel_sha256="orig_hash",
        checkpoint_sha256="ckpt_hash",
    )

    masks = Masks(ink_mask=ink_mask)
    res = reconstruct(orig, cf, masks=masks, reconstruction_config=ReconstructionConfig(export_size="original"))

    orig_arr = np.asarray(orig)
    res_arr = np.asarray(res.image)
    ink_pixels = np.asarray(ink_mask) > 128
    diff = np.abs(orig_arr[ink_pixels].astype(int) - res_arr[ink_pixels].astype(int))
    assert np.max(diff) == 0, "AT09 failed: ink pixels were altered"


def test_out01_non_overwriting_export_and_report(tmp_path):
    """OUT01: export_result rejects existing destination when overwrite=False and saves audit report."""
    orig = Image.new("RGB", (32, 32), (180, 180, 180))
    rgb_c = np.ones((16, 16, 3), dtype=np.float32) * 0.5
    cf = ColorField(
        rgb_canonical=rgb_c,
        lab_canonical=srgb_to_cielab(rgb_c),
        canonical_size=(16, 16),
        pad=(0, 0, 0, 0),
        original_size=orig.size,
        cache_key="cf_export_test",
        original_pixel_sha256="orig",
        checkpoint_sha256="ckpt",
    )
    res = reconstruct(orig, cf)

    out_file = tmp_path / "result.png"
    record = export_result(res, out_file)
    assert out_file.exists()
    assert record.report_path.exists()

    report_data = json.loads(record.report_path.read_text(encoding="utf-8"))
    assert report_data["sha256"] == record.sha256
    assert report_data["dimensions"] == [32, 32]

    # Re-exporting to same path must fail with FileExistsError (OUT01)
    with pytest.raises(FileExistsError, match="OUT01"):
        export_result(res, out_file, export_config=ExportConfig(overwrite=False))
