import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest

from mangacolor.projects import (
    CharacterPart,
    CharacterProfile,
    Page,
    Project,
    Region,
)


def test_character_profile_serialization(tmp_path):
    """CON01: Save and load character profile following schema mangacolor.character.v1."""
    profile = CharacterProfile(
        character_id="hero_01",
        name="Protagonista",
        revision=2,
        parts={
            "hair": CharacterPart(base_srgb="#285DAD", mode="base_color"),
            "skin": CharacterPart(base_srgb="#DBA98A", mode="base_color"),
            "jacket": CharacterPart(base_srgb="#A63835", mode="exact_rgb"),
        },
        references=["ref1.png"],
    )

    path = tmp_path / "hero.json"
    profile.save_json(path)
    assert path.exists()

    loaded = CharacterProfile.load_json(path)
    assert loaded.character_id == "hero_01"
    assert loaded.revision == 2
    assert loaded.parts["hair"].base_srgb == "#285DAD"
    assert loaded.parts["jacket"].mode == "exact_rgb"
    assert loaded.schema == "mangacolor.character.v1"


def test_project_multi_region_priority_hint_materialization(tmp_path):
    """CON02: Materialize page hints from character parts respecting priority order."""
    proj_dir = tmp_path / "project"
    masks_dir = proj_dir / "masks"
    masks_dir.mkdir(parents=True)

    # Create two overlapping masks for 50x50 page
    # Mask 1: Hair (covers rectangle 10,10 to 40,40)
    m1 = Image.new("L", (50, 50), 0)
    ImageDraw.Draw(m1).rectangle((10, 10, 40, 40), fill=255)
    m1.save(masks_dir / "hair.png")

    # Mask 2: Hat (covers rectangle 20,20 to 30,30, overlays hair with higher priority)
    m2 = Image.new("L", (50, 50), 0)
    ImageDraw.Draw(m2).rectangle((20, 20, 30, 30), fill=255)
    m2.save(masks_dir / "hat.png")

    char = CharacterProfile(
        character_id="char_1",
        name="Personaje",
        parts={
            "hair": CharacterPart(base_srgb="#0000FF"),  # Blue
            "hat": CharacterPart(base_srgb="#FF0000"),   # Red
        },
    )

    page = Page(
        page_id="p1",
        original="assets/p1.png",
        pixel_sha256="fake_sha",
        width=50,
        height=50,
        regions=[
            Region(
                region_id="r1",
                mask="masks/hair.png",
                character_id="char_1",
                part_id="hair",
                profile_revision=1,
                priority=100,
            ),
            Region(
                region_id="r2",
                mask="masks/hat.png",
                character_id="char_1",
                part_id="hat",
                profile_revision=1,
                priority=200,  # Higher priority must win in overlap
            ),
        ],
    )

    proj = Project(
        project_id="test_proj",
        pages=[page],
        characters={"char_1": char},
        root_dir=proj_dir,
    )
    proj.save()

    hints = proj.materialize_page_hints("p1")
    assert hints is not None
    assert hints.size == (50, 50)
    hints_arr = np.asarray(hints)

    # Pixel (15, 15) is hair -> Blue [0, 0, 255, 255]
    np.testing.assert_array_equal(hints_arr[15, 15], [0, 0, 255, 255])

    # Pixel (25, 25) is overlapping hat -> Red [255, 0, 0, 255] (higher priority wins!)
    np.testing.assert_array_equal(hints_arr[25, 25], [255, 0, 0, 255])

    # Pixel (5, 5) has no region -> Transparent [0, 0, 0, 0]
    np.testing.assert_array_equal(hints_arr[5, 5], [0, 0, 0, 0])


def test_character_update_invalidates_cache_con03(tmp_path):
    """CON03: Updating a character's color increments revision and invalidates affected page caches."""
    char = CharacterProfile(
        character_id="c1",
        name="Protagonista",
        revision=1,
        parts={"hair": CharacterPart(base_srgb="#112233")},
    )
    page = Page(
        page_id="p1",
        original="p1.png",
        pixel_sha256="sha",
        width=100,
        height=100,
        cached_color_field="predictions/p1_canon.npz",
        regions=[Region("r1", "mask.png", "c1", "hair", profile_revision=1)],
    )
    proj = Project(project_id="test", pages=[page], characters={"c1": char}, root_dir=tmp_path)

    assert page.cached_color_field is not None

    # Update hair color
    proj.update_character_part("c1", "hair", "#445566")

    # Verification: character revision incremented and page cache invalidated
    assert char.revision == 2
    assert char.parts["hair"].base_srgb == "#445566"
    assert page.cached_color_field is None
