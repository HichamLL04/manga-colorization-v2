"""Strict, weights-only checkpoint I/O for native and controlled v2 models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import warnings
import zipfile

import torch
from torch import Tensor

from .model import ControlledColorizer, NativeGenerator


FORMAT = "mangacolor.controlled.v1"
ADAPTER_PREFIX = "palette_adapter."


def _state_dict(value: object, name: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty tensor state_dict")
    if any(not isinstance(k, str) or not isinstance(v, Tensor) for k, v in value.items()):
        raise ValueError(f"{name} must map parameter names to tensors")
    return value


def _reject_incompatible(state: Mapping[str, Tensor]) -> None:
    for key in ("to0.0.weight", "generator.to0.0.weight"):
        weight = state.get(key)
        if weight is not None and weight.ndim == 4 and weight.shape[1] == 6:
            raise ValueError(
                "This is a v1 generator (6 input channels). It cannot be loaded "
                "as a v2 generator (5 input channels); use v2generator.zip."
            )
    if "to0.0.weight" not in state:
        raise ValueError(
            "Not a native v2 generator state_dict: missing to0.0.weight. "
            "Extractor, denoiser, v1 and TorchScript checkpoints are not supported."
        )
    first = state["to0.0.weight"]
    if tuple(first.shape) != (32, 5, 3, 3):
        raise ValueError(
            "Incompatible generator input: expected to0.0.weight shape "
            f"(32, 5, 3, 3), got {tuple(first.shape)}"
        )


def _read(path: str | Path) -> object:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # torch.load historically auto-dispatched TorchScript archives to jit.load.
    # Explicitly reject them before deserialization to keep a weights-only path.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if any("/code/" in name or name.endswith("/constants.pkl") for name in archive.namelist()):
                raise ValueError(
                    "TorchScript archive detected. Use the native v2 state_dict "
                    "v2generator.zip; colorizer.zip is a different model format."
                )
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch build lacks weights_only loading; install the project's "
            "supported PyTorch version. Unsafe pickle fallback is disabled."
        ) from exc


def native_state_dict(model: NativeGenerator) -> OrderedDict[str, Tensor]:
    """Return CPU tensors using precisely the original v2 parameter names."""
    return OrderedDict(
        (key, value.detach().cpu())
        for key, value in model.state_dict().items()
        if not key.startswith(ADAPTER_PREFIX)
    )


def load_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    allow_untrained_palette: bool = False,
) -> tuple[ControlledColorizer, dict]:
    """Load an exact v2 base or controlled checkpoint; return eval model, metadata.

    Unknown/missing/unexpected weights fail strict loading.  No prefix stripping,
    partial load, random replacement, or unsafe pickle retry is performed.
    ``allow_untrained_palette`` is for initializing a training run.
    """
    payload = _read(path)
    if isinstance(payload, Mapping) and "format" in payload:
        if payload["format"] != FORMAT:
            raise ValueError(f"Unsupported checkpoint format: {payload['format']!r}")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("checkpoint metadata must be a dictionary")
        metadata = dict(raw_metadata)
        if type(metadata.get("palette_trained", False)) is not bool:
            raise ValueError("metadata palette_trained must be a boolean")
        state = _state_dict(payload.get("generator"), "generator")
        adapter = _state_dict(payload.get("palette_adapter"), "palette_adapter")
        metadata["format"] = FORMAT
        metadata["epoch"] = payload.get("epoch", 0)
        metadata["step"] = payload.get("step", 0)
    else:
        state = _state_dict(payload, "native generator")
        adapter = None
        metadata = {"format": "mangacolor.native.v2", "palette_trained": False}
    _reject_incompatible(state)
    # Validate the base independently so an adapter cannot overwrite its keys.
    base = NativeGenerator()
    base.load_state_dict(state, strict=True)
    model = ControlledColorizer(allow_untrained_palette=allow_untrained_palette)
    complete = OrderedDict(model.state_dict())
    complete.update(state)
    model.load_state_dict(complete, strict=True)
    if adapter is not None:
        model.palette_adapter.load_state_dict(adapter, strict=True)
    metadata.setdefault("palette_trained", False)
    model.palette_trained = metadata["palette_trained"]
    model.to(device)
    model.eval()
    return model, metadata


def save_checkpoint(
    path: str | Path,
    model: ControlledColorizer,
    metadata: Mapping,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
) -> None:
    """Save a controlled checkpoint, keeping original v2 and adapter weights apart.

    The caller must explicitly record ``palette_trained=True`` after actually
    performing palette-conditioned optimization.  Merely adding this module or
    loading a native checkpoint never marks the palette as trained.
    """
    info = dict(metadata)
    if type(info.get("palette_trained", False)) is not bool:
        raise ValueError("metadata palette_trained must be a boolean")
    info.setdefault("palette_trained", False)
    payload = {
        "format": FORMAT,
        "generator": native_state_dict(model),
        "palette_adapter": OrderedDict(
            (key, value.detach().cpu()) for key, value in model.palette_adapter.state_dict().items()
        ),
        "metadata": info,
        "epoch": int(epoch),
        "step": int(step),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def export_native(path: str | Path, model: NativeGenerator) -> None:
    """Export a strict upstream v2 generator; global palette control is omitted."""
    warnings.warn(
        "Native v2 export contains generator weights and local-hint support. "
        "The global palette adapter is omitted; retain the controlled checkpoint "
        "to use global palettes.",
        UserWarning,
        stacklevel=2,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(native_state_dict(model), path)


__all__ = ["FORMAT", "export_native", "load_model", "native_state_dict", "save_checkpoint"]
