"""Behavioral checks against the supplied, real upstream v2 checkpoint."""

from pathlib import Path

import pytest
import torch

from mangacolor.checkpoints import export_native, load_model, native_state_dict, save_checkpoint
from mangacolor.model import ControlledColorizer, NativeGenerator, PaletteAdapter


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def real_v2():
    path = ROOT / "v2generator.zip"
    if not path.exists():
        pytest.skip("Supply the upstream v2generator.zip to run checkpoint integration checks")
    return load_model(path)


def test_real_v2_matches_original_exactly_without_palette(real_v2):
    controlled, metadata = real_v2
    original = NativeGenerator().eval()
    state = torch.load(ROOT / "v2generator.zip", map_location="cpu", weights_only=True)
    original.load_state_dict(state, strict=True)
    sketch = torch.rand(1, 5, 64, 64)
    sketch[:, 1:] = 0
    with torch.inference_mode():
        expected = original(sketch)
        actual = controlled(sketch)
    assert metadata["palette_trained"] is False
    assert set(native_state_dict(controlled)) == set(state)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_palette_adapter_is_order_invariant_and_masks_padding():
    adapter = PaletteAdapter()
    # Exercise a nonzero learned projection, rather than only its zero init.
    torch.nn.init.normal_(adapter.projection.weight, std=0.1)
    torch.nn.init.normal_(adapter.projection.bias, std=0.1)
    palette = torch.rand(2, 4, 3) * 2 - 1
    mask = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]])
    permutation = torch.tensor([2, 0, 3, 1])
    actual = adapter(palette, mask)
    reordered = adapter(palette[:, permutation], mask[:, permutation])
    torch.testing.assert_close(actual, reordered)
    assert torch.count_nonzero(actual[1]).item() == 0
    changed_padding = palette.clone()
    changed_padding[mask == 0] = -changed_padding[mask == 0]
    torch.testing.assert_close(actual, adapter(changed_padding, mask), rtol=0, atol=0)
    assert torch.count_nonzero(adapter(palette[:, :0])).item() == 0


def test_zero_initialized_adapter_preserves_native_output(real_v2):
    model, _ = real_v2
    previous = model.allow_untrained_palette
    model.allow_untrained_palette = True
    try:
        sketch = torch.rand(1, 5, 32, 32)
        palette = torch.tensor([[[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]]])
        with torch.inference_mode():
            baseline = model(sketch)
            conditioned = model(sketch, palette)
        for actual, expected in zip(conditioned, baseline):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        model.allow_untrained_palette = previous


def test_native_checkpoint_rejects_untrained_global_palette(real_v2):
    model, _ = real_v2
    with pytest.raises(RuntimeError, match="no trained global palette adapter"):
        model(torch.zeros(1, 5, 32, 32), torch.zeros(1, 1, 3))


def test_native_export_strict_roundtrip(real_v2, tmp_path):
    model, _ = real_v2
    path = tmp_path / "generator.zip"
    with pytest.warns(UserWarning, match="adapter is omitted"):
        export_native(path, model)
    exported = torch.load(path, map_location="cpu", weights_only=True)
    original = NativeGenerator()
    original.load_state_dict(exported, strict=True)
    assert not any(key.startswith("palette_adapter.") for key in exported)
    for key, expected in native_state_dict(model).items():
        torch.testing.assert_close(exported[key], expected, rtol=0, atol=0)


def test_controlled_checkpoint_roundtrip_keeps_palette_separate(real_v2, tmp_path):
    model, _ = real_v2
    path = tmp_path / "controlled.pt"
    save_checkpoint(path, model, {"palette_trained": True, "dataset": "test"}, epoch=3, step=5)
    loaded, metadata = load_model(path)
    assert loaded.palette_trained is True
    assert metadata["epoch"] == 3
    assert metadata["step"] == 5
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert "palette_adapter" in payload
    assert not any(key.startswith("palette_adapter.") for key in payload["generator"])
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected, rtol=0, atol=0)


def test_palette_adapter_receives_gradient_through_real_generator(real_v2):
    pretrained, _ = real_v2
    model = ControlledColorizer().eval()
    model.load_state_dict(pretrained.state_dict(), strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("palette_adapter."))
    sketch = torch.rand(1, 5, 32, 32)
    sketch[:, 1:] = 0
    palette = torch.tensor([[[0.8, -0.7, 0.2], [-0.3, 0.1, 0.9]]])
    prediction, _ = model(sketch, palette)
    prediction.square().mean().backward()
    gradient = model.palette_adapter.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_v1_and_torchscript_are_rejected(tmp_path):
    incompatible = tmp_path / "v1.pt"
    torch.save({"to0.0.weight": torch.zeros(32, 6, 3, 3)}, incompatible)
    with pytest.raises(ValueError, match="v1 generator"):
        load_model(incompatible)
    import zipfile

    script = tmp_path / "script.zip"
    with zipfile.ZipFile(script, "w") as archive:
        archive.writestr("colorizer/code/__torch__.py", "untrusted content")
    with pytest.raises(ValueError, match="TorchScript archive detected"):
        load_model(script)


def test_missing_native_parameter_fails_strict_load(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save({"to0.0.weight": torch.zeros(32, 5, 3, 3)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)
