"""Controllable manga colorization, with the original v2 as its backbone."""

__version__ = "0.2.0"

from .inference import ColorizationPipeline, open_rgb, parse_palette
from .reconstruction import (
    ColorField,
    Conditions,
    ExportConfig,
    ExportRecord,
    InferenceConfig,
    Masks,
    ReconstructionConfig,
    RenderResult,
    export_result,
    predict_color_field,
    reconstruct,
)
from .projects import (
    CharacterPart,
    CharacterProfile,
    Page,
    Project,
    Region,
)

__all__ = [
    "ColorizationPipeline",
    "open_rgb",
    "parse_palette",
    "ColorField",
    "Conditions",
    "ExportConfig",
    "ExportRecord",
    "InferenceConfig",
    "Masks",
    "ReconstructionConfig",
    "RenderResult",
    "export_result",
    "predict_color_field",
    "reconstruct",
    "CharacterPart",
    "CharacterProfile",
    "Page",
    "Project",
    "Region",
]

