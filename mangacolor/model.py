"""The original v2 generator, extended with an optional trainable palette adapter.

The five spatial input channels retain the upstream contract: grayscale [0, 1],
masked RGB hints [-1, 1], and hint mask [0, 1].  Global palettes use RGB [-1, 1].
No change is made to the names or shapes of the original generator parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .vendor.v2.models import Generator as NativeGenerator


class PaletteAdapter(nn.Module):
    """A masked DeepSets encoder producing FiLM parameters for v2's x0.

    Palette entries are a set, so their order must not change the result.  The
    final projection starts at zero: adding this adapter to a pretrained v2
    initially reproduces its output exactly, even when a palette is supplied.
    An empty palette always produces zero modulation, including after training.
    """

    def __init__(self, channels: int = 32, hidden: int = 64):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.projection = nn.Linear(hidden, channels * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, palette: Tensor, palette_mask: Tensor | None = None) -> Tensor:
        if palette.ndim != 3 or palette.shape[-1] != 3:
            raise ValueError("palette must have shape [batch, colors, 3]")
        if not palette.is_floating_point():
            raise ValueError("palette must contain floating-point RGB in [-1, 1]")
        if not torch.isfinite(palette).all() or (palette.abs() > 1.00001).any():
            raise ValueError("palette RGB must be finite and normalized to [-1, 1]")
        batch, count, _ = palette.shape
        if palette_mask is None:
            palette_mask = palette.new_ones((batch, count))
        if palette_mask.shape != (batch, count):
            raise ValueError("palette_mask must have shape [batch, colors]")
        if not torch.isfinite(palette_mask).all():
            raise ValueError("palette_mask must contain finite values")
        if ((palette_mask != 0) & (palette_mask != 1)).any():
            raise ValueError("palette_mask entries must be 0 or 1")
        weights = palette_mask.to(device=palette.device, dtype=palette.dtype)
        active = weights.sum(dim=1, keepdim=True)
        features = self.encoder(palette)
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / active.clamp_min(1)
        # Multiplication *after* projection also gates its learned bias.
        return self.projection(pooled) * (active > 0).to(pooled.dtype)


class ControlledColorizer(NativeGenerator):
    """Checkpoint-compatible v2 with learned global palette conditioning.

    Construction permits training a new adapter.  ``load_model`` disables
    untrained palette use by default; local hints remain available with any v2
    checkpoint.  ``palette_trained`` describes training provenance, not a
    guarantee that a particular checkpoint follows a palette well.
    """

    def __init__(self, ngf: int = 64, *, allow_untrained_palette: bool = True):
        super().__init__(ngf=ngf)
        self.palette_adapter = PaletteAdapter(channels=32)
        self.palette_trained = False
        self.allow_untrained_palette = allow_untrained_palette

    def forward(
        self,
        sketch: Tensor,
        palette: Tensor | None = None,
        palette_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if sketch.ndim != 4 or sketch.shape[1] != 5:
            raise ValueError("sketch must have shape [batch, 5, height, width]")
        if sketch.shape[2] % 32 or sketch.shape[3] % 32:
            raise ValueError("sketch height and width must be multiples of 32")
        if palette is None and palette_mask is not None:
            raise ValueError("palette_mask requires a palette")

        x0 = self.to0(sketch)
        if palette is not None:
            if palette.ndim != 3 or palette.shape[-1] != 3:
                raise ValueError("palette must have shape [batch, colors, 3]")
            if palette.shape[0] != sketch.shape[0]:
                raise ValueError("palette and sketch batch sizes must match")
            if not self.palette_trained and not self.allow_untrained_palette:
                raise RuntimeError(
                    "This checkpoint has no trained global palette adapter. "
                    "Use local color hints, train the adapter, or load a "
                    "controlled checkpoint with metadata palette_trained=true."
                )
            palette = palette.to(device=x0.device, dtype=x0.dtype)
            modulation = self.palette_adapter(palette, palette_mask)
            scale, shift = modulation.chunk(2, dim=1)
            x0 = x0 * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        # Preserve the upstream computation and all native state_dict keys.
        aux_out = self.to1(x0)
        aux_out = self.to2(aux_out)
        aux_out = self.to3(aux_out)
        x1, x2, x3, x4 = self.encoder(sketch[:, 0:1])
        out = self.tunnel4(torch.cat([x4, aux_out], 1))
        x = self.tunnel3(torch.cat([out, x3], 1))
        x = self.tunnel2(torch.cat([x, x2, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        decoder_output = self.deconv_for_decoder(out)
        return x, decoder_output


__all__ = ["ControlledColorizer", "NativeGenerator", "PaletteAdapter"]
