import importlib
import pathlib
import sys
import types

import numpy as np
import pytest

from scripts import test_single_image as single_image


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_detects_supported_model_types() -> None:
    assert single_image.model_type(pathlib.Path("model.onnx")) == "onnx"
    assert single_image.model_type(pathlib.Path("model.keras")) == "keras"


def test_rejects_unsupported_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported model format"):
        single_image.model_type(pathlib.Path("model.pt"))


def test_resolves_adjacent_plate_config(tmp_path: pathlib.Path) -> None:
    model_path = tmp_path / "best.onnx"
    config_path = tmp_path / "plate_config.yaml"
    model_path.touch()
    config_path.touch()

    assert single_image.resolve_plate_config(model_path, None) == config_path.resolve()


def test_rejects_missing_plate_config(tmp_path: pathlib.Path) -> None:
    model_path = tmp_path / "best.keras"
    model_path.touch()

    with pytest.raises(FileNotFoundError, match=str(tmp_path / "plate_config.yaml")):
        single_image.resolve_plate_config(model_path, None)


@pytest.mark.parametrize(
    ("onnx_type", "expected"),
    [("tensor(uint8)", np.dtype(np.uint8)), ("tensor(float)", np.dtype(np.float32))],
)
def test_casts_input_from_onnx_metadata(onnx_type: str, expected: np.dtype) -> None:
    batch = np.array([[[[255]]]], dtype=np.uint8)

    assert single_image.cast_onnx_input(batch, onnx_type).dtype == expected


def test_rejects_unsupported_onnx_input_type() -> None:
    batch = np.array([[[[255]]]], dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported ONNX input type"):
        single_image.cast_onnx_input(batch, "tensor(float16)")


def test_selects_named_plate_output() -> None:
    expected = np.ones((1, 8, 3), dtype=np.float32)

    assert np.array_equal(single_image.plate_output({"plate": expected}), expected)


def test_selects_first_sequence_output() -> None:
    expected = np.ones((1, 8, 3), dtype=np.float32)

    assert np.array_equal(single_image.plate_output([expected]), expected)


def test_rejects_empty_output_sequence() -> None:
    with pytest.raises(ValueError, match="no outputs"):
        single_image.plate_output([])


def test_selects_onnx_providers() -> None:
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    assert single_image.select_onnx_providers("auto", available) == available
    assert single_image.select_onnx_providers("cpu", available) == ["CPUExecutionProvider"]
    assert single_image.select_onnx_providers("cuda", available) == ["CUDAExecutionProvider"]


def test_rejects_unavailable_onnx_provider() -> None:
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is not available"):
        single_image.select_onnx_providers("cuda", ["CPUExecutionProvider"])


def test_runs_onnx_with_metadata_dtype(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    captured: dict[str, object] = {}
    expected = np.ones((1, 8, 3), dtype=np.float32)

    class FakeSession:
        def __init__(self, model_path: pathlib.Path, providers: list[str]) -> None:
            captured["model_path"] = model_path
            captured["providers"] = providers

        @staticmethod
        def get_inputs() -> list[types.SimpleNamespace]:
            return [types.SimpleNamespace(name="input", type="tensor(float)")]

        @staticmethod
        def get_outputs() -> list[types.SimpleNamespace]:
            return [types.SimpleNamespace(name="plate")]

        @staticmethod
        def run(output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            captured["output_names"] = output_names
            captured["input"] = feeds["input"]
            return [expected]

    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    batch = np.ones((1, 2, 3, 1), dtype=np.uint8)
    model_path = tmp_path / "best.onnx"

    actual = single_image.run_onnx(batch, model_path, "cpu")

    assert np.array_equal(actual, expected)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["output_names"] == ["plate"]
    assert captured["input"].dtype == np.float32  # type: ignore[union-attr]


def test_runs_keras_with_requested_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    captured: dict[str, object] = {}
    expected = np.ones((1, 8, 3), dtype=np.float32)

    class FakeModel:
        def __call__(self, batch: np.ndarray, training: bool) -> dict[str, np.ndarray]:
            captured["batch"] = batch
            captured["training"] = training
            return {"plate": expected}

    def load_model(model_path: pathlib.Path, compile: bool) -> FakeModel:
        captured["model_path"] = model_path
        captured["compile"] = compile
        return FakeModel()

    imports: list[str] = []

    def record_import(name: str) -> None:
        imports.append(name)

    monkeypatch.setitem(
        sys.modules, "keras", types.SimpleNamespace(models=types.SimpleNamespace(load_model=load_model))
    )
    monkeypatch.setattr(importlib, "import_module", record_import)
    batch = np.ones((1, 2, 3, 1), dtype=np.uint8)
    model_path = tmp_path / "best.keras"

    actual = single_image.run_keras(batch, model_path, "torch")

    assert np.array_equal(actual, expected)
    assert imports == ["fast_plate_ocr.train.model.layers"]
    assert captured["training"] is False
    assert captured["compile"] is False


def test_loads_and_resizes_one_image() -> None:
    batch, config = single_image.load_batch(
        PROJECT_ROOT / "test" / "assets" / "test_plate_1.png",
        PROJECT_ROOT / "trained_models" / "cblprd_cct_s_v2_torch" / "2026-07-27_11-57-15" / "plate_config.yaml",
    )

    assert batch.shape == (1, 64, 128, 3)
    assert batch.dtype == np.uint8
    assert config.max_plate_slots == 8


def test_decodes_plate_output() -> None:
    config = types.SimpleNamespace(max_plate_slots=2, alphabet="AB_", pad_char="_")
    output = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)

    assert single_image.decode_plate(output, config) == "AB"


def test_main_prints_model_type_and_plate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image_path = tmp_path / "plate.png"
    model_path = tmp_path / "best.onnx"
    config_path = tmp_path / "plate_config.yaml"
    for path in (image_path, model_path, config_path):
        path.touch()
    batch = np.ones((1, 2, 3, 1), dtype=np.uint8)
    config = types.SimpleNamespace()

    def fake_load_batch(_image: pathlib.Path, _plate_config: pathlib.Path) -> tuple[np.ndarray, object]:
        return batch, config

    def fake_run_onnx(_data: np.ndarray, _model: pathlib.Path, _device: str) -> np.ndarray:
        return np.ones((1, 1, 1))

    def fake_decode_plate(_output: np.ndarray, _plate_config: object) -> str:
        return "京A12345"

    monkeypatch.setattr(single_image, "load_batch", fake_load_batch)
    monkeypatch.setattr(single_image, "run_onnx", fake_run_onnx)
    monkeypatch.setattr(single_image, "decode_plate", fake_decode_plate)

    return_code = single_image.main(
        [
            "--image",
            str(image_path),
            "--model",
            str(model_path),
            "--plate-config",
            str(config_path),
            "--device",
            "cpu",
        ]
    )

    assert return_code == 0
    assert capsys.readouterr().out.splitlines() == ["model_type=onnx", "plate=京A12345"]


def test_main_reports_missing_image(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_path = tmp_path / "best.onnx"
    config_path = tmp_path / "plate_config.yaml"
    model_path.touch()
    config_path.touch()

    return_code = single_image.main(
        [
            "--image",
            str(tmp_path / "missing.png"),
            "--model",
            str(model_path),
            "--plate-config",
            str(config_path),
        ]
    )

    assert return_code == 1
    assert "Image not found" in capsys.readouterr().err
