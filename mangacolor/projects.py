"""Domain models, projects, character profiles and region links.

Implements MangaColor SSD Milestone C (CON01 to CON05, Section 8 and Section 12):
- CharacterProfile: persistent chromatic profiles (schema mangacolor.character.v1)
- Project: multi-page project management (schema mangacolor.project.v1)
- Region: manual and explicit mask-to-character-part binding with priority
- Materialization of consistent page hints across manga panels
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .color import srgb_to_linear
from .reconstruction import (
    ColorField,
    Conditions,
    ExportConfig,
    ExportRecord,
    InferenceConfig,
    Masks,
    ReconstructionConfig,
    RenderResult,
    compute_image_pixels_sha256,
    compute_sha256,
    export_result,
    predict_color_field,
    reconstruct,
)


@dataclass
class CharacterPart:
    """Color definition for a specific anatomical or clothing part."""
    base_srgb: str  # Hex #RRGGBB
    mode: str = "base_color"  # "base_color" (preserves shading) or "exact_rgb"

    def to_rgb_tuple(self) -> Tuple[int, int, int]:
        clean = self.base_srgb.strip().lstrip("#")
        if len(clean) != 6:
            raise ValueError(f"Color hex inválido: {self.base_srgb}")
        return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)


@dataclass
class CharacterProfile:
    """Persistent character chromatic profile (schema mangacolor.character.v1)."""
    character_id: str
    name: str
    revision: int = 1
    parts: Dict[str, CharacterPart] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    schema: str = "mangacolor.character.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "character_id": self.character_id,
            "name": self.name,
            "revision": self.revision,
            "parts": {k: asdict(v) for k, v in self.parts.items()},
            "references": list(self.references),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterProfile":
        if data.get("schema") != "mangacolor.character.v1":
            raise ValueError(f"Esquema de personaje no reconocido: {data.get('schema')}")
        parts = {
            k: CharacterPart(**v) if isinstance(v, dict) else v
            for k, v in data.get("parts", {}).items()
        }
        return cls(
            character_id=data["character_id"],
            name=data["name"],
            revision=data.get("revision", 1),
            parts=parts,
            references=data.get("references", []),
            schema=data.get("schema", "mangacolor.character.v1"),
        )

    def save_json(self, path: Union[str, Path]) -> Path:
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return dest

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "CharacterProfile":
        source = Path(path).resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class Region:
    """A masked region linked to a character part (SSD Section 8.3 & 12.2)."""
    region_id: str
    mask: str  # Path relative to project root
    character_id: str
    part_id: str
    profile_revision: int
    mode: str = "base_color"
    priority: int = 100  # Higher priority overlays lower priority

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Region":
        return cls(**data)


@dataclass
class Page:
    """A page within a manga colorization project."""
    page_id: str
    original: str  # Path relative to project root
    pixel_sha256: str
    width: int
    height: int
    regions: List[Region] = field(default_factory=list)
    text_mask: Optional[str] = None  # Path relative to project root
    ink_mask: Optional[str] = None   # Path relative to project root
    cached_color_field: Optional[str] = None  # Path to canonical .npz
    last_rendered_revision: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["regions"] = [r.to_dict() if isinstance(r, Region) else r for r in self.regions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Page":
        regions = [Region.from_dict(r) if isinstance(r, dict) else r for r in data.get("regions", [])]
        data_copy = dict(data)
        data_copy["regions"] = regions
        return cls(**data_copy)


@dataclass
class Project:
    """MangaColor project management (schema mangacolor.project.v1)."""
    project_id: str
    revision: int = 1
    color_space: str = "sRGB"
    pages: List[Page] = field(default_factory=list)
    characters: Dict[str, CharacterProfile] = field(default_factory=dict)
    render_config: Dict[str, Any] = field(default_factory=lambda: {
        "canonical_size": 576,
        "detail_strength": 1.0,
        "alpha_original": 0.0,
    })
    schema: str = "mangacolor.project.v1"
    root_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "project_id": self.project_id,
            "revision": self.revision,
            "color_space": self.color_space,
            "pages": [p.to_dict() for p in self.pages],
            "characters": [c.to_dict() for c in self.characters.values()],
            "render_config": self.render_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], root_dir: Optional[Path] = None) -> "Project":
        if data.get("schema") != "mangacolor.project.v1":
            raise ValueError(f"Esquema de proyecto no reconocido: {data.get('schema')}")
        pages = [Page.from_dict(p) for p in data.get("pages", [])]
        characters = {
            c["character_id"]: CharacterProfile.from_dict(c)
            for c in data.get("characters", [])
        }
        return cls(
            project_id=data["project_id"],
            revision=data.get("revision", 1),
            color_space=data.get("color_space", "sRGB"),
            pages=pages,
            characters=characters,
            render_config=data.get("render_config", {}),
            schema=data.get("schema", "mangacolor.project.v1"),
            root_dir=root_dir,
        )

    def save(self, directory: Optional[Union[str, Path]] = None) -> Path:
        target_dir = Path(directory or self.root_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir = target_dir
        dest = target_dir / "project.json"
        dest.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return dest

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "Project":
        dir_path = Path(directory).resolve()
        proj_file = dir_path / "project.json"
        if not proj_file.exists():
            raise FileNotFoundError(f"No se encontró project.json en {dir_path}")
        data = json.loads(proj_file.read_text(encoding="utf-8"))
        return cls.from_dict(data, root_dir=dir_path)

    def get_page(self, page_id: str) -> Page:
        for p in self.pages:
            if p.page_id == page_id:
                return p
        raise KeyError(f"Página '{page_id}' no encontrada en el proyecto.")

    def add_character(self, profile: CharacterProfile) -> None:
        self.characters[profile.character_id] = profile
        self.revision += 1

    def update_character_part(self, character_id: str, part_id: str, base_srgb: str, mode: str = "base_color") -> None:
        """Update a part color and increment character revision, invalidating affected pages (CON03)."""
        if character_id not in self.characters:
            raise KeyError(f"Personaje '{character_id}' no existe.")
        char = self.characters[character_id]
        char.parts[part_id] = CharacterPart(base_srgb=base_srgb, mode=mode)
        char.revision += 1
        self.revision += 1
        # Invalidate page cache for any page using this character
        for page in self.pages:
            if any(r.character_id == character_id for r in page.regions):
                page.cached_color_field = None

    def materialize_page_hints(self, page_id: str) -> Optional[Image.Image]:
        """Rasterize character-linked regions into an RGBA hints image (SSD 12.2).

        Regions are composited by explicit priority (higher priority on top).
        Ties are broken by order of insertion.
        """
        page = self.get_page(page_id)
        if not page.regions:
            return None

        width, height = page.width, page.height
        # RGBA canvas initialized to transparent
        canvas = np.zeros((height, width, 4), dtype=np.uint8)

        # Sort regions by priority ascending so highest priority draws last
        sorted_regions = sorted(page.regions, key=lambda r: r.priority)

        for region in sorted_regions:
            if region.character_id not in self.characters:
                continue
            char = self.characters[region.character_id]
            if region.part_id not in char.parts:
                continue
            part = char.parts[region.part_id]
            r, g, b = part.to_rgb_tuple()

            # Load mask
            base = self.root_dir or Path(".")
            mask_file = (base / region.mask).resolve()
            if not mask_file.exists():
                continue
            with Image.open(mask_file) as mask_img:
                m = mask_img.convert("L")
                if m.size != (width, height):
                    m = m.resize((width, height), Image.Resampling.NEAREST)
                mask_arr = np.asarray(m)

            active = mask_arr > 128
            canvas[active, 0] = r
            canvas[active, 1] = g
            canvas[active, 2] = b
            canvas[active, 3] = 255

        return Image.fromarray(canvas, mode="RGBA")

    def render_page(
        self,
        page_id: str,
        pipeline: Any,
        export_size: Union[str, int, Tuple[int, int]] = "original",
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
    ) -> ExportRecord:
        """Render page enforcing character consistency and decoupled reconstruction (CON01-CON04)."""
        page = self.get_page(page_id)
        base = self.root_dir or Path(".")
        orig_file = (base / page.original).resolve()

        from .inference import open_rgb
        original_img = open_rgb(orig_file)

        hints_img = self.materialize_page_hints(page_id)

        # Check if canonical field is already cached and valid
        cf = None
        if page.cached_color_field:
            cf_file = (base / page.cached_color_field).resolve()
            if cf_file.exists():
                cf = ColorField.load_npz(cf_file)

        if cf is None:
            canonical_size = self.render_config.get("canonical_size", 576)
            conditions = Conditions(hints=hints_img, palette=[], palette_mode="hints")
            inf_cfg = InferenceConfig(canonical_size=canonical_size, device=str(pipeline.device))
            cf = predict_color_field(original_img, conditions, pipeline.model, inf_cfg, pipeline.metadata)

            # Persist canonical field to predictions cache
            pred_dir = base / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            cf_path = pred_dir / f"{page.page_id}_canon.npz"
            cf.save_npz(cf_path)
            page.cached_color_field = str(cf_path.relative_to(base))

        # Masks
        text_m = None
        if page.text_mask:
            text_file = (base / page.text_mask).resolve()
            if text_file.exists():
                text_m = Image.open(text_file).convert("L")

        ink_m = None
        if page.ink_mask:
            ink_file = (base / page.ink_mask).resolve()
            if ink_file.exists():
                ink_m = Image.open(ink_file).convert("L")

        # Locked hints if any region has mode exact_rgb
        has_exact = any(r.mode == "exact_rgb" for r in page.regions)
        masks = Masks(
            text_mask=text_m,
            ink_mask=ink_m,
            hints_lock=hints_img,
            lock_hints=has_exact,
        )

        recon_cfg = ReconstructionConfig(
            export_size=export_size,
            detail_strength=self.render_config.get("detail_strength", 1.0),
            alpha_original=self.render_config.get("alpha_original", 0.0),
        )

        render_res = reconstruct(original_img, cf, masks=masks, reconstruction_config=recon_cfg)

        out_base = Path(output_dir) if output_dir else (base / "exports")
        out_base.mkdir(parents=True, exist_ok=True)
        dest_img = out_base / f"{page.page_id}_export.png"

        rec = export_result(render_res, dest_img, ExportConfig(overwrite=overwrite, save_report=True))
        page.last_rendered_revision = self.revision
        self.save()
        return rec
