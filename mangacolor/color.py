"""High-precision color space conversions and gamut mapping.

Implements sRGB <-> Linear sRGB <-> CIE XYZ (D65) <-> CIELAB as specified
in MangaColor SSD Section 11.2 and 11.5.
"""
from typing import Tuple
import numpy as np


# Standard sRGB D65 transformation matrix (Y=1.0)
# Reference: IEC 61966-2-1 / MangaColor SSD Section 11.2
M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)

M_XYZ_TO_SRGB = np.linalg.inv(M_SRGB_TO_XYZ).astype(np.float32)

# D65 standard white point
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

# CIELAB constants
DELTA = 6.0 / 29.0
DELTA_CUBE = DELTA ** 3
THREE_DELTA_SQ = 3.0 * (DELTA ** 2)


def validate_finite(array: np.ndarray, name: str = "array") -> None:
    """Ensure array has no NaNs or Infs."""
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contiene valores no finitos (NaN o Inf).")


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0, 1] to linear RGB in [0, 1]."""
    validate_finite(rgb, "rgb")
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((np.maximum(rgb, 0.0) + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB in [0, 1]."""
    validate_finite(linear, "linear")
    linear = np.asarray(linear, dtype=np.float32)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (np.maximum(linear, 0.0) ** (1.0 / 2.4)) - 0.055,
    ).astype(np.float32)


def linear_rgb_to_xyz(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to CIE XYZ (D65)."""
    return np.matmul(linear, M_SRGB_TO_XYZ.T).astype(np.float32)


def xyz_to_linear_rgb(xyz: np.ndarray) -> np.ndarray:
    """Convert CIE XYZ (D65) to linear RGB."""
    return np.matmul(xyz, M_XYZ_TO_SRGB.T).astype(np.float32)


def xyz_to_cielab(xyz: np.ndarray) -> np.ndarray:
    """Convert CIE XYZ (D65) to CIELAB.

    Returns L* in [0, 100], and signed a*, b*.
    """
    validate_finite(xyz, "xyz")
    t = xyz / WHITE_D65
    f = np.where(
        t > DELTA_CUBE,
        np.cbrt(np.maximum(t, 0.0)),
        t / THREE_DELTA_SQ + (4.0 / 29.0),
    )
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def cielab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to CIE XYZ (D65)."""
    validate_finite(lab, "lab")
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def f_inv(u: np.ndarray) -> np.ndarray:
        return np.where(u > DELTA, u ** 3, THREE_DELTA_SQ * (u - 4.0 / 29.0))

    x = WHITE_D65[0] * f_inv(fx)
    y = WHITE_D65[1] * f_inv(fy)
    z = WHITE_D65[2] * f_inv(fz)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def srgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    """Direct conversion from sRGB [0, 1] to CIELAB."""
    linear = srgb_to_linear(rgb)
    xyz = linear_rgb_to_xyz(linear)
    return xyz_to_cielab(xyz)


def cielab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Direct conversion from CIELAB to unclipped sRGB."""
    xyz = cielab_to_xyz(lab)
    linear = xyz_to_linear_rgb(xyz)
    return linear_to_srgb(linear)


def map_gamut(lab: np.ndarray, iterations: int = 10, tolerance: float = 0.01) -> Tuple[np.ndarray, np.ndarray, float]:
    """Gamut mapping towards neutral axis at constant lightness.

    Reduces chroma gradually until the pixel falls inside sRGB [0, 1].
    Any minor numerical residual (< tolerance) is safely clipped to [0, 1].

    Returns:
        (mapped_srgb, mapped_lab, affected_fraction)
    """
    validate_finite(lab, "lab")
    L = np.clip(lab[..., 0:1], 0.0, 100.0)
    ab = lab[..., 1:3]

    initial_srgb = cielab_to_srgb(lab)
    in_gamut = (initial_srgb >= -tolerance) & (initial_srgb <= 1.0 + tolerance)
    all_in = np.all(in_gamut, axis=-1)

    out_mask = ~all_in
    affected_count = int(np.count_nonzero(out_mask))
    total_pixels = lab.shape[0] * lab.shape[1] if lab.ndim >= 2 else lab.shape[0]
    affected_fraction = float(affected_count / max(1, total_pixels))

    # In-gamut pixels preserve 100% of their chroma (factor = 1.0)
    s_factor = np.ones(lab.shape[:-1], dtype=np.float32)

    if np.any(out_mask):
        L_out = L[out_mask]
        ab_out = ab[out_mask]
        low = np.zeros(L_out.shape[:-1], dtype=np.float32)
        high = np.ones(L_out.shape[:-1], dtype=np.float32)

        for _ in range(iterations):
            mid = 0.5 * (low + high)
            test_ab = ab_out * mid[..., None]
            test_lab = np.concatenate([L_out, test_ab], axis=-1)
            test_srgb = cielab_to_srgb(test_lab)
            valid = np.all((test_srgb >= -tolerance) & (test_srgb <= 1.0 + tolerance), axis=-1)
            low = np.where(valid, mid, low)
            high = np.where(valid, high, mid)

        s_factor[out_mask] = low

    final_ab = ab * s_factor[..., None]
    final_lab = np.concatenate([L, final_ab], axis=-1)
    final_srgb = np.clip(cielab_to_srgb(final_lab), 0.0, 1.0)
    return final_srgb, final_lab, affected_fraction
