"""Guided image filtering and smooth filtering for resolution decoupling.

Implements Guided Filter (He, Sun, Tang 2010) for chrominance upsampling,
with support for seamless tile-based processing (AT10) and smooth luminance
decomposition (SSD Section 11.3 & 11.4).
"""
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F


def _box_filter_torch(x: torch.Tensor, r: int) -> torch.Tensor:
    """Fast box filter using replicate padding and average pooling."""
    kernel_size = 2 * r + 1
    # Replicate padding keeps boundary values stable and prevents darkening
    x_pad = F.pad(x, (r, r, r, r), mode="replicate")
    return F.avg_pool2d(x_pad, kernel_size, stride=1, padding=0)


def guided_filter_torch(guide: torch.Tensor,
                        src: torch.Tensor,
                        r: int = 4,
                        eps: float = 1e-3) -> torch.Tensor:
    """Guided Filter (He et al. 2010) on PyTorch tensors.

    Args:
        guide: Guidance tensor of shape (B, 1, H, W), normalized float32.
        src: Input tensor to filter, shape (B, C, H, W), float32.
        r: Filter radius.
        eps: Regularization parameter.

    Returns:
        Filtered tensor of shape (B, C, H, W).
    """
    if r < 1:
        return src

    mean_I = _box_filter_torch(guide, r)
    mean_p = _box_filter_torch(src, r)
    corr_I = _box_filter_torch(guide * guide, r)
    corr_Ip = _box_filter_torch(guide * src, r)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter_torch(a, r)
    mean_b = _box_filter_torch(b, r)

    q = mean_a * guide + mean_b
    return q


def guided_filter_tiled_torch(guide: torch.Tensor,
                              src: torch.Tensor,
                              r: int = 4,
                              eps: float = 1e-3,
                              tile_size: int = 1024,
                              margin: Optional[int] = None) -> torch.Tensor:
    """Tiled Guided Filter with context margin to avoid seams (AT10).

    A context margin >= 2*r ensures exact mathematical equivalence with
    full-image filtering.
    """
    _, _, H, W = guide.shape
    if H <= tile_size and W <= tile_size:
        return guided_filter_torch(guide, src, r=r, eps=eps)

    if margin is None:
        margin = max(16, 2 * r)

    output = torch.zeros_like(src)

    for y in range(0, H, tile_size):
        y_end = min(H, y + tile_size)
        y0_pad = max(0, y - margin)
        y1_pad = min(H, y_end + margin)

        for x in range(0, W, tile_size):
            x_end = min(W, x + tile_size)
            x0_pad = max(0, x - margin)
            x1_pad = min(W, x_end + margin)

            tile_guide = guide[:, :, y0_pad:y1_pad, x0_pad:x1_pad]
            tile_src = src[:, :, y0_pad:y1_pad, x0_pad:x1_pad]

            filtered_tile = guided_filter_torch(tile_guide, tile_src, r=r, eps=eps)

            # Slice out the valid interior region (without context padding)
            crop_y0 = y - y0_pad
            crop_y1 = crop_y0 + (y_end - y)
            crop_x0 = x - x0_pad
            crop_x1 = crop_x0 + (x_end - x)

            output[:, :, y:y_end, x:x_end] = filtered_tile[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

    return output


def guided_filter_np(guide: np.ndarray,
                     src: np.ndarray,
                     r: int = 4,
                     eps: float = 1e-3,
                     tile_size: int = 1024) -> np.ndarray:
    """Numpy interface for Guided Filter."""
    guide_t = torch.from_numpy(guide).float()
    src_t = torch.from_numpy(src).float()

    if guide_t.ndim == 2:
        guide_t = guide_t.unsqueeze(0).unsqueeze(0)
    elif guide_t.ndim == 3:
        guide_t = guide_t.permute(2, 0, 1).unsqueeze(0)

    if src_t.ndim == 2:
        src_t = src_t.unsqueeze(0).unsqueeze(0)
    elif src_t.ndim == 3:
        src_t = src_t.permute(2, 0, 1).unsqueeze(0)

    filtered = guided_filter_tiled_torch(guide_t, src_t, r=r, eps=eps, tile_size=tile_size)
    res = filtered.squeeze(0)
    if res.ndim == 3 and res.shape[0] > 1:
        return res.permute(1, 2, 0).cpu().numpy()
    return res.squeeze(0).cpu().numpy()


def smooth_filter_np(image: np.ndarray, radius: int = 5, filter_type: str = "box_v1") -> np.ndarray:
    """Smooth luminance filter (filtro_suave_versionado) from SSD Section 11.4."""
    img_t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    if filter_type == "box_v1":
        smoothed = _box_filter_torch(img_t, radius)
    elif filter_type == "gaussian_v1":
        kernel_size = 2 * radius + 1
        sigma = max(0.5, radius / 2.0)
        coords = torch.arange(kernel_size, dtype=torch.float32) - radius
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = g / g.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)
        pad = radius
        padded = F.pad(img_t, (pad, pad, pad, pad), mode="replicate")
        smoothed = F.conv2d(padded, kernel_2d)
    else:
        raise ValueError(f"Tipo de filtro suave desconocido: {filter_type}")

    return smoothed.squeeze().cpu().numpy()
